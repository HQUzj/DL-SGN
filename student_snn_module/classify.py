
nsteps = 16
import torchvision.transforms as transforms
from .classifylayer.layers import *
import torchvision
import torch
class Image_classifier(nn.Module):  # Example net for MNIST
    def __init__(self):
        super(Image_classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, 5, 1, 2, bias=None)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(15, 40, 5, 1, 2, bias=None)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 40, 300)
        self.fc2 = nn.Linear(300, 10)

        self.conv1_s = tdLayer(self.conv1)
        self.pool1_s = tdLayer(self.pool1)
        self.conv2_s = tdLayer(self.conv2)
        self.pool2_s = tdLayer(self.pool2)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2_s = tdLayer(self.fc2)

        self.spike = LIFSpike()
        self.steps, _, _ = get_snn_param()

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.pool1_s(x)
        x = self.spike(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.pool2_s(x)
        x = self.spike(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / self.steps  # [N, neurons, steps]
        return out


class My_Classifier_mnist(nn.Module):  # Example net for MNIST
    def __init__(self):
        super(My_Classifier_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, 5, 1, 2, bias=None)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(15, 40, 5, 1, 2, bias=None)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 40, 300)
        self.fc2 = nn.Linear(300, 4)

        self.conv1_s = tdLayer(self.conv1)
        self.pool1_s = tdLayer(self.pool1)
        self.conv2_s = tdLayer(self.conv2)
        self.pool2_s = tdLayer(self.pool2)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2_s = tdLayer(self.fc2)

        self.spike = LIFSpike()
        self.steps, _, _ = get_snn_param()

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.pool1_s(x)
        x = self.spike(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.pool2_s(x)
        x = self.spike(x)
        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / self.steps  # [N, neurons, steps]
        return out

class My_Classifier_mnist_3d(nn.Module):  # Example net for MNIST
    def __init__(self):
        super(My_Classifier_mnist_3d, self).__init__()
        self.conv1 = nn.Conv2d(3, 15, 5, 1, 2, bias=None)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(15, 40, 5, 1, 2, bias=None)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 40, 300)
        self.fc2 = nn.Linear(300, 10)

        self.conv1_s = tdLayer(self.conv1)
        self.pool1_s = tdLayer(self.pool1)
        self.conv2_s = tdLayer(self.conv2)
        self.pool2_s = tdLayer(self.pool2)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2_s = tdLayer(self.fc2)

        self.spike = LIFSpike()
        self.steps, _, _ = get_snn_param()

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.spike(x)
        x = self.pool1_s(x)
        x = self.spike(x)
        x = self.conv2_s(x)
        x = self.spike(x)
        x = self.pool2_s(x)
        x = self.spike(x)
        x = x.view(x.shape[0], -1, x.shape[4])#64,2560,16
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)
        x = self.spike(x)
        out = torch.sum(x, dim=2) / self.steps  # [N, neurons, steps]
        return out




class ANN_My_Classifier_mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.len_discrete_code = 4
        self.main = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=4,stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.linear_main = nn.Sequential(
            nn.Linear(128*6*6,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.out_logit = nn.Linear(64,self.len_discrete_code)
        self.softmaxValue = nn.Softmax()
    def forward(self,x):
        x = self.main(x)
        x = x.reshape(x.shape[0],-1,)
        x = self.linear_main(x)
        out_logit = self.out_logit(x)
        softmaxValue = self.softmaxValue(out_logit)
        return out_logit, softmaxValue





def crition(logit,labels):
    # logit = logit.long()
    bc_loss =  nn.functional.binary_cross_entropy_with_logits(logit,labels)
    return bc_loss

class ANN_Image_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,padding=1),
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





data_path='../data'
dis_batch_size = 64
num_workers = 0
SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
ds = torchvision.datasets.MNIST(data_path, train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.Resize(32),
                                              transforms.ToTensor(),
                                              SetRange
                                          ]))
train_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                                  shuffle=True, drop_last=True, pin_memory=True,
                                                  num_workers=num_workers)



ds = torchvision.datasets.FashionMNIST(data_path, train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.Resize(32),
                                              transforms.ToTensor(),
                                              SetRange
                                          ]))
fstrain_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                                  shuffle=True, drop_last=True, pin_memory=True,
                                                  num_workers=num_workers)



ds = torchvision.datasets.MNIST(data_path, train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.Resize(32),
                                              transforms.ToTensor(),
                                              SetRange
                                          ]))
test_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                                  shuffle=True, drop_last=True, pin_memory=True,
                                                  num_workers=num_workers)

ds = torchvision.datasets.FashionMNIST(data_path, train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.Resize(32),
                                              transforms.ToTensor(),
                                              SetRange
                                          ]))
fs_test_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                                  shuffle=True, drop_last=True, pin_memory=True,
                                                  num_workers=num_workers)


def ANN_Calculate_accuracy(predict_networks,test_loader,device):
    correct = torch.zeros(1).squeeze().cuda(device)
    total = torch.zeros(1).squeeze().cuda(device)
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        label_logits = predict_networks(images)
        label_softmax = F.softmax(label_logits)
        predictions = torch.argmax(label_softmax, dim=1)
        correct += (predictions == labels).sum().float()
        total += len(labels)
    acc_str = ((correct / total).cpu().detach().data.numpy())
    return acc_str
def Calculate_accuracy(predict_networks,test_loader,device):
    correct = torch.zeros(1).squeeze().cuda(device)
    total = torch.zeros(1).squeeze().cuda(device)
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        if images.shape[1] == 1:
            images = images.unsqueeze(-1).repeat(1, 3, 1, 1, nsteps)
        else:
            images = images.unsqueeze(-1).repeat(1, 1, 1, 1, nsteps)
        labels = labels.to(device)
        label_logits = predict_networks(images)
        # label_softmax = F.softmax(label_logits)
        predictions = torch.argmax(label_logits, dim=1)
        correct += (predictions == labels).sum().float()
        total += len(labels)
    acc_str = ((correct / total).cpu().detach().data.numpy())
    return acc_str

device =0
model = My_Classifier_mnist_3d().cuda(1)
optimizer = torch.optim.Adam(model.parameters(),
                                    lr=0.0001,)
def test():
    for epoch in range(100):
        # for batch_idx, (data, target) in enumerate(train_loader):
        for batch_idx, (fs_img, mnist_img) in enumerate(zip(fstrain_loader, train_loader)):
            data, target = data.to(device), target.to(device)

            # necessary for general dataset: broadcast input
            # torch.zeros((16) + data.size())
            # data, _ = torch.broadcast_tensors(data, torch.zeros((16) + data.shape))
            # data = data.permute(1, 2, 3, 4, 0)
            data = data.unsqueeze(-1).repeat(1, 1, 1, 1, nsteps)
            # print(data.shape)
            output = model(data)
            # print(output[0])
            loss = F.cross_entropy(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch:{epoch}-batch_idex:{batch_idx} loss:{loss.item()}")
        print(Calculate_accuracy(model,test_loader,0))



def mnist_test():
    acc = 0
    for epoch in range(100):
        # for batch_idx, (data, target) in enumerate(train_loader):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # necessary for general dataset: broadcast input
            # torch.zeros((16) + data.size())
            # data, _ = torch.broadcast_tensors(data, torch.zeros((16) + data.shape))
            # data = data.permute(1, 2, 3, 4, 0)
            data = data.unsqueeze(-1).repeat(1, 3, 1, 1, nsteps)
            # print(data.shape)
            output = model(data)
            # print(output[0])
            loss = F.cross_entropy(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch:{epoch}-batch_idex:{batch_idx} loss:{loss.item()}")
        accu = Calculate_accuracy(model,test_loader,1)
        if accu > acc:
            torch.save(model.state_dict(),f"./3dmnist-{accu}-{epoch}.pth")
# mnist_test()

# test()
# test()

def test1():
    for epoch in range(100):
        # for batch_idx, (data, target) in enumerate(train_loader):
        for batch_idx, (fs_img, mnist_img) in enumerate(zip(fstrain_loader, train_loader)):
            shuffle = torch.randperm(128)
            # print(imgs[0].shape)
            # print(imgs[1].shape)
            fs_image = fs_img[0].to(device)
            minist_image = mnist_img[0].to(device)

            # print(fs_image.shape)
            # print(minist_image.shape)

            real_img = torch.cat([fs_image, 2 * minist_image - 1], dim=0)
            real_img = real_img[shuffle]
            gan_fs = (fs_image + 1) / 2
            gan_real_img = torch.cat([gan_fs, minist_image], dim=0)
            # print(real_imgs.shape)
            fs_label = fs_img[1].to(device)
            minist_label = mnist_img[1].to(device)
            label = torch.cat([fs_label, minist_label], dim=0)
            image_label = F.one_hot(label, 10)
            image_label = image_label[shuffle]
            image_label = image_label.float()
            # direct spike input

            spike_input = real_img.unsqueeze(-1).repeat(1, 1, 1, 1, nsteps)  # (N,C,H,W,T)
            output = model(spike_input)
            loss = F.cross_entropy(output, image_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch:{epoch}-batch_idex:{batch_idx} loss:{loss.item()}")
        print("mnist:",Calculate_accuracy(model,test_loader,0))
        print("mnist:",Calculate_accuracy(model,fs_test_loader,0))

# test1()

ann_classify = ANN_Image_classifier().cuda()
ann_optimizer = torch.optim.Adam(ann_classify.parameters(),
                                    lr=0.0001,)
def ann_test():
    best = 0
    for epoch in range(500):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target = F.one_hot(target,10)
            output = ann_classify(data)
            loss = crition(output, target.float())
            ann_optimizer.zero_grad()
            loss.backward()
            ann_optimizer.step()
            # print(loss)

        if epoch % 10 == 0:
            acc = ANN_Calculate_accuracy(ann_classify, test_loader, 0)
            if acc > best:
                print(f"epoch:{epoch}, acc:{acc}")
                torch.save(ann_classify.state_dict(),"best.pth")
# ann_test()