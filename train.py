from utils.eval import *
from student_snn_module.student_initialization import *
from assistant_module.assistant_initialization import *
from expert_module.wgan_gp import Expert
import datasets.load_dataset_snn as lds
import torch
import numpy as np
import random
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import copy
#Load dataset
taskcount=3
data_path = './dataset'
##task1 mnist
batch_size = 64
threshold = 0.015
mnist_train_loader, mnist_test_loader, mnist_dataset, _= lds.load_mnist(data_path, batch_size)
##task2 svhn
svhn_train_loader, svhn_test_loader, svhn_dataset, _ = lds.load_svhn(data_path, batch_size)
##task3 fmnist
fmnist_train_loader, fmnist_test_loader, fmnist_dataset, _ = lds.load_fashionmnist(data_path, batch_size)

#use assistant
def learning_with_assistant(imgs,label_one_hot,taskid):
    #update student vae
    student_vae.train()
    dual_encoder.train()
    curr_bs = imgs.shape[0]
    real_imgs = imgs.cuda(device)
    spike_input = real_imgs.unsqueeze(-1).repeat(1, 1, 1, 1, 16)  # (N,C,H,W,T)
    # ---------------------
    #  Train Discriminator
    # ---------------------
    for d_iter in range(5):
        opt_shared.zero_grad()
        opt_disc_head.zero_grad()
        z = student_vae.fsencoder.prior.sample(real_imgs.shape[0])
        real_validity = dual_encoder(real_imgs, mode="dis")
        fake_imgs = student_vae.fsdecoder(z, label_one_hot, taskid).detach()
        fake_validity = dual_encoder(fake_imgs, mode="dis")
        x_recon, q_z, p_z, sampled_z = student_vae(spike_input, label_one_hot, taskid,
                                                   scheduled=True)  # sampled_z(B,C,1,1,T)

        rec_validity = dual_encoder(x_recon, mode="dis")
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1.0 + fake_validity)) * 0.5 + \
                 torch.mean(nn.ReLU(inplace=True)(1.0 + rec_validity)) * 0.5
        d_loss.backward()
        opt_shared.step()
        opt_disc_head.step()

    # -----------------
    #  Train Generator
    # -----------------

    student_vae_encoder_optimizer.zero_grad()
    student_vae_decoder_optimizer.zero_grad()
    # gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (gen_batch_size, latent_dim)))
    # gen_imgs = decoder(gen_z)
    gen_z = student_vae.fsencoder.prior.sample(real_imgs.shape[0])
    gen_z = gen_z.detach()
    gen_imgs = student_vae.fsdecoder(gen_z, label_one_hot, taskid)
    fake_validity = dual_encoder(gen_imgs, mode="dis")
    # rec, mu, logvar = f_recon(real_imgs, encoder, decoder, latent_dim)
    x_recon, q_z, p_z, sampled_z = student_vae(spike_input, label_one_hot, taskid,
                                       scheduled=True)  # sampled_z(B,C,1,1,T)
    rec_validity = dual_encoder(x_recon, mode="dis")
    # cal loss
    if epoch == -1:
        g_loss = -(torch.mean(rec_validity) * 0.5)
    else:
        g_loss = -(torch.mean(fake_validity) * 0.5 + torch.mean(rec_validity) * 0.5)
    kld = 1 * student_vae.loss_function_mmd(real_imgs, x_recon, q_z, p_z)['Distance_Loss']
    (g_loss + kld).backward()
    student_vae_encoder_optimizer.step()
    student_vae_decoder_optimizer.step()

    # contrastive
    student_vae_encoder_optimizer.zero_grad()
    student_vae_decoder_optimizer.zero_grad()
    opt_shared.zero_grad()
    opt_cont_head.zero_grad()
    x_recon, q_z, p_z, sampled_z = student_vae(spike_input, label_one_hot, taskid,
                                       scheduled=True)  # sampled_z(B,C,1,1,T)
    im_k = real_imgs
    im_q = x_recon
    with torch.no_grad():
        # update momentum encoder
        for p, p_mom in zip(dual_encoder.parameters(), dual_encoder_M.parameters()):
            p_mom.data = (p_mom.data * 0.999) + (p.data * (1.0 - 0.999))
        d_k = dual_encoder_M(im_k, mode="cont")
        for l in layers:
            d_k[l] = F.normalize(d_k[l], dim=1)
    total_cont = torch.tensor(0.0).cuda(device)
    d_q = dual_encoder(im_q, mode="cont")
    for l in layers:
        q = F.normalize(d_q[l], dim=1)
        k = d_k[l]
        queue = d_queue[l]
        l_pos = torch.einsum("nc,nc->n", [k, q]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, queue.detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / cont_temp  # 0.07
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(device)
        cont_loss = nn.CrossEntropyLoss()(logits, labels) * lambda_cont
        cont_loss = cont_loss.cuda(device)
        total_cont = total_cont + cont_loss
    kld = 1 * student_vae.loss_function_mmd(real_imgs, x_recon, q_z, p_z)['Distance_Loss']
    (total_cont + kld).backward()
    student_vae_encoder_optimizer.step()
    student_vae_decoder_optimizer.step()
    opt_shared.step()
    opt_cont_head.step()

    for l in layers:
        ptr = int(d_queue_ptr[l])
        d_queue[l][:, ptr:(ptr + curr_bs)] = d_k[l].transpose(0, 1)
        ptr = (ptr + curr_bs) % cont_k  # move the pointer ahead
        d_queue_ptr[l][0] = ptr

    # update student classifier,the steps of classifier T can be configured separately.
    # print(spike_input.shape)
    output = student_classifier(spike_input)
    clable = torch.squeeze(label_one_hot[:, :, :1])
    loss = F.cross_entropy(output,clable)
    loss.backward()
    student_classifier_optimizer.zero_grad()

    return g_loss,(total_cont + kld),loss



def learning_without_assistant(imgs,label_one_hot,taskid):
    student_vae.train()
    student_vae.fsencoder.update_p(epoch, max_epoch)
    dual_encoder.train()
    curr_bs = imgs.shape[0]
    real_imgs = imgs.cuda(device)
    spike_input = real_imgs.unsqueeze(-1).repeat(1, 1, 1, 1, 16)  # (N,C,H,W,T)
    x_recon, q_z, p_z, sampled_z = student_vae(spike_input, label_one_hot, taskid,
                                               scheduled=True)
    student_vae_encoder_optimizer.zero_grad()
    student_vae_decoder_optimizer.zero_grad()
    loss = student_vae.loss_function_mmd(real_imgs, x_recon, q_z, p_z)['Loss']
    loss.backward()
    student_vae_encoder_optimizer.step()
    student_vae_decoder_optimizer.step()

    # update student classifier,the steps of classifier T can be configured separately.
    output = student_classifier(imgs)
    loss = F.cross_entropy(output, label_one_hot)
    loss.backward()
    student_classifier_optimizer.zero_grad()

def check_expansion_with_DKAF(experts,taskIndex,num_size):
    minDistance = 1e8
    expert_id = -1
    # 获取最后一个FSVAE
    if taskIndex == 1:
        dataset = svhn_dataset
    elif taskIndex == 2:
        dataset = fmnist_dataset

    # 获取 trainset 中所有数据点的索引
    all_indices = list(range(len(dataset)))

    # 随机选择 1000 个索引
    random_indices = random.sample(all_indices, num_size)

    # 创建一个空的张量来存储图像
    real_images = torch.empty(num_size, 3, 32, 32)  # 创建一个大小为 (1000, 1, 32, 32) 的空张量
    real_images = real_images.cuda(device)
    # 根据随机选择的索引获取图像并存储为张量
    for i, index in enumerate(random_indices):
        image, label = dataset[index]
        # 将 PIL 图像转换为 PyTorch 张量，并将其存储在 tensor_images 中
        # image = transforms.ToPILImage()(image)
        # real_images[i] = transforms.ToTensor()(image)
        real_images[i] = image.cuda(device)

    for i in range (len(experts)):
        z = torch.randn(num_size, 100, 1, 1).cuda(device)
        fake_images = experts[i].G(z)
        distance = experts[i].fake_wasserstein_distance(real_images, fake_images)
        if distance < minDistance:
            minDistance = distance
            expert_id = i

    return  minDistance, expert_id


def generate_fake_images(expert, num_samples, batch_size=64):
    """
    生成伪图像并返回一个数据加载器。

    Args:
        expert (list): 包含生成器 experts[i].G 和分类器 experts[i].Classifier。
        num_samples (int): 要生成的伪图像数量。
        batch_size (int): 每个小批次的大小。

    Returns:
        DataLoader: 包含生成的伪图像的数据加载器。
    """
    # 创建一个空的张量来存储伪图像
    fake_datasets =[]
    fake_dataloaders=[]
    with torch.no_grad():
        fake_images = torch.empty(num_samples, 1, 32, 32)  # 创建一个大小为 (num_samples, 1, 32, 32) 的空张量

        # 使用 experts 生成伪图像
        for i in range(0, num_samples, batch_size):
            # 生成一个小批次的伪图像
            batch_size_remaining = min(batch_size, num_samples - i)
            noise = torch.randn(batch_size_remaining, 100, 1, 1)  # 随机噪声输入
            generated_images = expert.G(noise)  # 使用生成器生成图像
            fake_images[i:i + batch_size_remaining] = generated_images

        # 将生成的伪图像转换为 PyTorch 数据集
        fake_dataset = TensorDataset(fake_images)

        # 创建数据加载器
        fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    return fake_loader



if __name__ == '__main__':
    ASSISTANT=True
    experts=[]
    for taskIndex in range(taskcount):
        if taskIndex == 0:
            current_snnData = mnist_train_loader
        elif taskIndex == 1:
            current_snnData = svhn_train_loader
        elif taskIndex == 2:
            current_snnData = fmnist_train_loader

        if taskIndex == 0: #
            expert = Expert(channels=3).cuda(device)
            for epoch in tqdm(range(max_epoch), desc='total progress'):
                for batch_idx, (real_img, label) in enumerate(current_snnData):

                    global_steps += 1

                    # spiking_label
                    real_img = real_img.cuda(device)
                    real_img = real_img.repeat(1,3,1,1)
                    labels = label.to(device, non_blocking=True)
                    label_one_hot = F.one_hot(labels, 10).float()
                    label_one_hot = label_one_hot.reshape([labels.shape[0], 10, 1])
                    label_one_hot = label_one_hot.repeat(1, 1, 16)
                    #spiking_taskid
                    taskid = torch.zeros(label_one_hot.shape).cuda(device)
                    taskid[:,taskIndex:taskIndex+2,:] = 1

                    ###Student  and Assistant learning
                    if ASSISTANT==True:
                        ass_loss, stu_loss, stu_closs =learning_with_assistant(imgs=real_img, label_one_hot=label_one_hot, taskid=taskid)
                    else:
                        learning_without_assistant(imgs=real_img,label_one_hot=label_one_hot,taskid=taskid)

                    # Expert Learning
                    d_loss,g_loss,c_loss = expert.expert_learning(images=real_img,label=labels)
                    # 使用字符串格式化打印损失值
                    print("Expert: d_loss: {:.4f}, g_loss: {:.4f}, c_loss: {:.4f} --Student: stu_loss: {:.4f}, stuc_loss: {:.4f} --Assistant: ass_loss: {:.4f}".format(d_loss, g_loss, c_loss, stu_loss, stu_closs, ass_loss))
                    if global_steps == 1:
                        break
                break
            experts.append(expert)
        else:
            if expert_id != -1: #add new expert
                for epoch in tqdm(range(max_epoch), desc='total progress'):
                    for batch_idx, (real_img, label) in enumerate(current_snnData):
                        # spiking_label
                        real_img = real_img.cuda(device)
                        labels = label.to(device, non_blocking=True)
                        real_img = real_img.repeat(1,3,1,1)
                        # spiking_taskid
                        taskid = torch.zeros(label_one_hot.shape).cuda(device)
                        taskid[:, taskIndex:taskIndex + 2, :] = 1
                        for i in range(len(experts)): #experts generate previous knowledge
                            fake_image = experts[i].fake_sample(batch_size=real_img.shape[0])
                            fake_taskid = torch.zeros(label_one_hot.shape).cuda(device)
                            if i == 0:
                                fake_taskid[:, i:i + 2, :] = 1
                            else:
                                fake_taskid[:, i:2 * i, :] = 1
                            fake_label = experts[i].Classifier(fake_image)
                            fake_label = torch.argmax(fake_label, dim=1)
                            real_img = torch.cat([real_img,fake_image],dim=0)
                            labels = torch.cat([labels,fake_label],dim=0)
                            taskid = torch.cat([taskid,fake_taskid],dim=0)
                        label_one_hot = F.one_hot(labels, 10).float()
                        label_one_hot = label_one_hot.reshape([labels.shape[0], 10, 1])
                        label_one_hot = label_one_hot.repeat(1, 1, 16)
                        ###Student  and Assistant learning
                        shuffle = torch.randperm(real_img.shape[0])
                        real_img = real_img[shuffle]
                        label_one_hot = label_one_hot[shuffle]
                        taskid = taskid[shuffle]
                        real_img = real_img[:,1:2,:,:]
                        if ASSISTANT == True:
                            ass_loss, stu_loss, stu_closs = learning_with_assistant(imgs=real_img, label_one_hot=label_one_hot, taskid=taskid)
                        else:
                            learning_without_assistant(imgs=real_img, label_one_hot=label_one_hot, taskid=taskid)

                        # Expert Learning
                        d_loss, g_loss, c_loss = expert.expert_learning(images=real_img, label=label)

                experts.append(expert)

            else: #select old expert
                old_expert = copy.deepcopy(experts[expert_id])
                for epoch in tqdm(range(max_epoch), desc='total progress'):
                    for batch_idx, (real_img, label) in enumerate(current_snnData):
                        bs = real_img.shape[0]
                        real_img = real_img.cuda(device)
                        if real_img.shape[1] == 1:
                            real_img = real_img.repeat([1,3,1,1])
                        # spiking_label
                        labels = label.to(device, non_blocking=True)
                        # spiking_taskid
                        taskid = torch.zeros([label.shape[0],10,16]).cuda(device)
                        taskid[:, taskIndex:taskIndex + 2, :] = 1

                        for i in range(len(experts)): #experts generate previous knowledge
                            fake_image = experts[i].fake_sample(batch_size=bs)
                            fake_taskid = torch.zeros(taskid.shape).cuda(device)
                            if i == 0:
                                fake_taskid[:, i:i + 2, :] = 1
                            else:
                                fake_taskid[:, i:2 * i, :] = 1
                            fake_label = experts[i].Classifier(fake_image)
                            fake_label = torch.argmax(fake_label, dim=1)
                            real_img = torch.cat([real_img,fake_image],dim=0)
                            label = torch.cat([labels,fake_label],dim=0)
                            taskid = torch.cat([taskid,fake_taskid],dim=0)

                        # selected expert generate image
                        old_image = old_expert.fake_sample(bs)
                        old_lable = old_expert.Classifier(old_image)
                        old_lable = torch.argmax(old_lable, dim=1)
                        old_taskid = torch.zeros([bs,10,16]).cuda(device)
                        old_taskid[:,expert_id:expert_id+2,:]=1

                        real_img = torch.cat([real_img,old_image],dim=0)
                        label = torch.cat([label,old_lable],dim=0)
                        taskid = torch.cat([taskid,old_taskid],dim=0)
                        label_one_hot = F.one_hot(label, 10).float()
                        label_one_hot = label_one_hot.reshape([label.shape[0], 10, 1])
                        label_one_hot = label_one_hot.repeat(1, 1, 16)


                        ###Student  and Assistant learning
                        shuffle = torch.randperm(taskid.shape[0])
                        real_img = real_img[shuffle]
                        # print(real_img.shape)
                        label_one_hot = label_one_hot[shuffle]
                        taskid = taskid[shuffle]
                        real_img = real_img[:,1:2,:,:]
                        ###Student  and Assistant learning
                        if ASSISTANT == True:
                            d_loss, g_loss, c_loss = learning_with_assistant(imgs=real_img, label_one_hot=label_one_hot, taskid=taskid)
                        else:
                            learning_without_assistant(imgs=real_img, label_one_hot=label_one_hot, taskid=taskid)

                fake_datasets, fake_dataloaders = generate_fake_images(expert, num_samples=50000)
                for epoch in tqdm(range(max_epoch), desc='total progress'):
                        # Expert Learning
                    for (data1, label1), (data2, label2) in zip(current_snnData, fake_dataloaders):
                        data1 = data1.cuda(device)
                        data2 = data2.cuda(device)
                        label1 = label1.cuda(device)
                        label2 = label2.cuda(device)
                        real_img = torch.cat([data1,data2],dim=0)
                        label = torch.cat([label1,label2], dim=0)
                        expert.expert_learning(images=real_img, label=label)

                experts.append(expert)
        nexttaskIndex = taskIndex + 1
        if nexttaskIndex < 3:
            minDistance, expert_id = check_expansion_with_DKAF(experts,nexttaskIndex,num_size=1000)
            if minDistance < threshold:
                expert = Expert(channels=3)
                expert_id = -1

            else:
                expert = experts[expert_id]

    # 保存模型列表到文件
    torch.save(experts, 'experts_list.pth')
    torch.save(student_vae,'stu_vae.pth')
    torch.save(student_classifier,'stu_classifer.pth')