
import torch
import torch.nn as nn
from .snn_layers import *
from .snnvae_prior import *
from .snnvae_posterior import *
import torch.nn.functional as F

from focal_frequency_loss import FocalFrequencyLoss as FFL
import torchvision
ffl = FFL(loss_weight=1.0, alpha=1.0)  # initialize nn.Module class

# import global_v as glv

bce_loss = nn.BCELoss(reduction='sum')


class FSencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = 0
        in_channels = 3
        # in_channels = 1
        latent_dim = 128
        self.latent_dim = latent_dim
        self.n_steps = 16

        self.k = 20

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        is_first_conv = True
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                       out_channels=h_dim,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       bias=True,
                       bn=tdBatchNorm(h_dim),
                       spike=LIFSpike(),
                       is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False

        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1] * 4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())
        self.prior =  PriorBernoulliSTBP(k=20)
        self.posterior = PosteriorBernoulliSTBP(k=20)
    def forward(self, x, scheduled=False):
        x = self.encoder(x)  # (N,C,H,W,T)
        x = torch.flatten(x, start_dim=1, end_dim=3)  # (N,C*H*W,T)
        latent_x = self.before_latent_layer(x)  # (N,latent_dim,T)
        sampled_z, q_z = self.posterior(latent_x)  # sampled_z:(B,C,1,1,T), q_z:(B,C,k,T)

        p_z = self.prior(sampled_z, scheduled, self.p)
        return sampled_z, q_z, p_z

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p - init_p) * epoch / max_epoch + init_p


class FSdecoder(nn.Module):
    def __init__(self):
        super().__init__()
        latent_dim = 128
        self.latent_dim = latent_dim
        self.n_steps = 16

        self.k = 20

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()



        # Build Decoder
        modules = []

        self.decoder_input = tdLinear(latent_dim +10+10,
                                      hidden_dims[-1] * 4,
                                      bias=True,
                                      bn=tdBatchNorm(hidden_dims[-1] * 4),
                                      spike=LIFSpike())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                tdConvTranspose(hidden_dims[i],
                                hidden_dims[i + 1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1,
                                bias=True,
                                bn=tdBatchNorm(hidden_dims[i + 1]),
                                spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            tdConvTranspose(hidden_dims[-1],
                            hidden_dims[-1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                            bias=True,
                            bn=tdBatchNorm(hidden_dims[-1]),
                            spike=LIFSpike()),
            tdConvTranspose(hidden_dims[-1],
                            # out_channels=1,
                            out_channels=3,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            bn=None,
                            spike=None)
        )



        self.membrane_output_layer = MembraneOutputLayer()


    def forward(self, z,cond,taskid):
        z = torch.cat([z, cond,taskid], dim=1)
        result = self.decoder_input(z)  # (N,C*H*W,T)
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, self.n_steps)  # (N,C,H,W,T)
        result = self.decoder(result)  # (N,C,H,W,T)
        result = self.final_layer(result)  # (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result))
        return out
device = 0
fsenc = FSencoder().cuda(device)
fsdec = FSdecoder().cuda(device)

class SNNVAE(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 3
        latent_dim = 128
        self.latent_dim = latent_dim
        self.n_steps = 16

        self.k = 20
        self.device=1

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        self.fsencoder = fsenc
        self.fsdecoder = fsdec
        self.p = 0
        self.psp = PSP()

    def forward(self, x, cond, taskid,scheduled=False):
        # sampled_z, q_z, p_z = self.encode(x, scheduled)
        sampled_z, q_z, p_z = self.fsencoder(x, scheduled)
        # x_recon = self.decode(sampled_z)
        x_recon = self.fsdecoder(sampled_z,cond ,taskid)
        return x_recon, q_z, p_z, sampled_z


    def sample(self, batch_size=64): #cond sample
        # sampled_z = self.prior.sample(batch_size)
        choice = torch.randint(low=0, high=10, size=(1, batch_size)).cuda(device)
        label = torch.nn.functional.one_hot(choice, 10)
        label = label.reshape([label.shape[1], 10, 1])
        label = label.repeat(1, 1, 16)

        sampled_z = self.fsencoder.prior.sample(batch_size)
        taskid = torch.ones(size=label.shape).cuda(device)
        taskid[:,0:5,:] = 0
        sampled_x = self.fsdecoder(sampled_z, label,taskid)
        return sampled_x, sampled_z

    def fsample(self, batch_size=64):  # cond sample
        # sampled_z = self.prior.sample(batch_size)
        choice = torch.randint(low=0, high=10, size=(1, batch_size)).cuda(device)
        label = torch.nn.functional.one_hot(choice, 10)
        label = label.reshape([label.shape[1], 10, 1])
        label = label.repeat(1, 1, 16)

        sampled_z = self.fsencoder.prior.sample(batch_size)
        taskid = torch.ones(size=label.shape).cuda(device)
        taskid[:, 5:10, :] = 0
        sampled_x = self.fsdecoder(sampled_z, label, taskid)
        return sampled_x, sampled_z


    def loss_function_mmd(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = ffl(recons_img,input_img)+F.mse_loss(recons_img, input_img)
        q_z_ber = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        p_z_ber = torch.mean(p_z, dim=2) # (N, latent_dim, T)

        #kld_loss = torch.mean((q_z_ber - p_z_ber)**2)
        mmd_loss = torch.mean((self.psp(q_z_ber)-self.psp(p_z_ber))**2)
        loss = recons_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Distance_Loss': mmd_loss}


    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4,4)

    # def update_p(self, epoch, max_epoch):
    #     init_p = 0.1
    #     last_p = 0.3
    #     self.p = (last_p-init_p) * epoch / max_epoch + init_p
    #

