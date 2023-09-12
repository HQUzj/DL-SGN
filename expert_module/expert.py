import torch
from wgan_gp import WGAN
class Expert(torch.nn.Module):
    def __init__(self,device):
        super(Expert, self).__init__()
        self.device = device
        self.wgan = expert_VAE().cuda(self.device)
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
    def expert_learning(self,image):
        inputs = image.reshape(image.shape[0], -1)
        recon, mu, log_std,feature = self.vae(inputs)
        loss = self.vae.loss_function(recon, inputs, mu, log_std)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 计算Wasserstein距离
    def wasserstein_distance(self, real, fake):
        recon, mu, log_std,feature1 = self.vae(real)
        recon, mu, log_std,feature2 = self.vae(fake)
        return torch.mean(feature1 - feature2)

    def check_expansion(self, students):
        # 获取最后一个FSVAE
        last_fsvae = students[-1].fsvae
        last_fake_samples = last_fsvae.sample(128)
        # torchvision.utils.save_image(last_fake_samples, f"student-{len(students)}.png")
        # last_fake_samples = (last_fake_samples+1)/2
        # print(last_fake_samples.shape)
        # 计算最小Wasserstein距离
        min_distance = 1e8
        for i in range(len(students) - 1):
            fsvae = students[i].fsvae
            real_samples = fsvae.sample(128)
            # real_samples = (real_samples + 1)/2.
            real_samples = real_samples.reshape(real_samples.shape[0], -1)
            last_fake_samples = last_fake_samples.reshape(last_fake_samples.shape[0], -1)
            distance = self.wasserstein_distance(real_samples, last_fake_samples)
            distance = torch.abs(distance)
            if distance < min_distance:
                min_distance = distance
        return min_distance