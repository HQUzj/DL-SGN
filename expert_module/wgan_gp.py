import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import torchvision
import torch.nn.functional as F

class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class ANN_Image_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5,padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,kernel_size=5,padding=1),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.linear_main = nn.Sequential(
            nn.Linear(2304,1024),
            nn.Dropout(),
            nn.Linear(1024,10),
        )
    def forward(self,x):
        x = self.main(x)
        x = self.linear_main(x)
        return x


class Expert(torch.nn.Module):
    def __init__(self, channels=3,device=0):
        super().__init__()
        self.G = Generator(channels).cuda(device)
        self.D = Discriminator(channels).cuda(device)
        self.Classifier = ANN_Image_classifier().cuda(device)
        self.device = device
        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 128

        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.c_optimizer = optim.Adam(self.Classifier.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.critic_iter = 3
        self.lambda_term = 10

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(real_images.shape[0],1,1,1).uniform_(0,1)
        eta = eta.expand(real_images.shape[0], real_images.size(1), real_images.size(2), real_images.size(3))
        if self.device>-1:
            eta = eta.cuda(self.device)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.device>-1:
            interpolated = interpolated.cuda(self.device)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.device) if self.device>-1 else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty


    def expert_learning(self,images, label):
        images = images.repeat(1,3,1,1)
        one = torch.tensor(1, dtype=torch.float)
        one = one.cuda(self.device)
        mone = one * -1
        # Requires grad, Generator requires_grad = False
        for p in self.D.parameters():
            p.requires_grad = True

        d_loss_real = 0
        d_loss_fake = 0
        Wasserstein_D = 0
        # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
        for d_iter in range(self.critic_iter):
            self.D.zero_grad()

            # Train discriminator
            # WGAN - Training discriminator more iterations than generator
            # Train with real images
            # print(images.shape)
            d_loss_real = self.D(images)
            d_loss_real = d_loss_real.mean()
            d_loss_real.backward(mone)

            # Train with fake images
            z = torch.rand((images.shape[0], 100, 1, 1))
            z = z.cuda(self.device)
            fake_images = self.G(z)
            d_loss_fake = self.D(fake_images)
            d_loss_fake = d_loss_fake.mean()
            d_loss_fake.backward(one)

            # Train with gradient penalty
            gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
            gradient_penalty.backward()

            d_loss = d_loss_fake - d_loss_real + gradient_penalty
            Wasserstein_D = d_loss_real - d_loss_fake
            self.d_optimizer.step()
            # print(
            #     f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

        # Generator update
        for p in self.D.parameters():
            p.requires_grad = False  # to avoid computation

        self.G.zero_grad()
        # train generator
        # compute loss with fake images
        z = torch.randn(self.batch_size, 100, 1, 1).cuda(self.device)
        fake_images = self.G(z)
        g_loss = self.D(fake_images)
        g_loss = g_loss.mean()
        g_loss.backward(mone)
        g_cost = -g_loss
        self.g_optimizer.step()

        #Classifier update
        output = self.Classifier(images)
        loss = F.cross_entropy(output, label)
        self.c_optimizer.zero_grad()
        loss.backward()
        self.c_optimizer.step()

        return d_loss,g_loss,loss
    # 计算Wasserstein距离
    def fake_wasserstein_distance(self, real, fake):
        real_features = self.D.feature_extraction(real)
        fake_features = self.D.feature_extraction(fake)
        return torch.mean(real_features - fake_features)



    def fake_sample(self,batch_size):
        noise = torch.randn(batch_size,100,1,1)
        noise = noise.cuda(self.device)
        fake_image = self.G(noise)
        return fake_image