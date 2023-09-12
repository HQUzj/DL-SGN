from .models_32 import *
import os
from utils.eval import *


#############################
# Assistant Hyperparameters
#############################
seed               = 123
# lr                 = 0.0002
lr                 = 0.0005
beta1              = 0.0
beta2              = 0.9
num_workers        = 0
data_path          = "dataset"

dis_batch_size     = 64 #64
gen_batch_size     = 128 #128
# max_epoch          = 800
max_epoch          = 200
lambda_kld         = 5e-4
latent_dim         = 128
cont_dim           = 16
cont_k             = 8192
cont_temp          = 0.07

# multi-scale contrastive setting
layers             = ["b1", "final"]

device = torch.device("cuda:0")
name =("").join(layers)
log_fname = f"logs/cifar10-{name}"
fid_fname = f"logs/FID_cifar10-{name}"
viz_dir = f"viz/cifar10-{name}"
models_dir = f"saved_models/cifar10-{name}"
if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
lambda_cont = 1.0/len(layers)
fix_seed(random_seed=seed)

#############################
# Make and initialize the Assistant Networks and Student Networks
#############################

# net = fsvae.SNNVAE().cuda(device)
dual_encoder = DualEncoder(cont_dim).cuda(device)
dual_encoder.apply(weights_init)
dual_encoder_M = DualEncoder(cont_dim).cuda(device)

for p, p_momentum in zip(dual_encoder.parameters(), dual_encoder_M.parameters()):
    p_momentum.data.copy_(p.data)
    p_momentum.requires_grad = False
# gen_avg_param = copy_params(net.fsdecoder)
d_queue, d_queue_ptr = {}, {}
for layer in layers:
    d_queue[layer] = torch.randn(cont_dim, cont_k).cuda(device)
    d_queue[layer] = F.normalize(d_queue[layer], dim=0)
    d_queue_ptr[layer] = torch.zeros(1, dtype=torch.long)


#############################
# Make the Assistant optimizers
#############################


shared_params = list(dual_encoder.block1.parameters()) + \
                list(dual_encoder.block2.parameters()) + \
                list(dual_encoder.block3.parameters()) + \
                list(dual_encoder.block4.parameters()) + \
                list(dual_encoder.l5.parameters())
opt_shared = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        shared_params),
                                0.2*lr, (beta1, beta2))
opt_disc_head = torch.optim.Adam(filter(lambda p: p.requires_grad,
                    dual_encoder.head_disc.parameters()),
                0.2*lr, (beta1, beta2))
cont_params = list(dual_encoder.head_b1.parameters()) + \
                list(dual_encoder.head_b2.parameters()) + \
                list(dual_encoder.head_b3.parameters()) + \
                list(dual_encoder.head_b4.parameters())
opt_cont_head = torch.optim.Adam(filter(lambda p: p.requires_grad, cont_params),
                0.2*lr, (beta1, beta2))


global_steps = 0
# train loop

#
