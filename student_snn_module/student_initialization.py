from .classify import *
from .snn_vae import *


device = 0
lr                 = 0.0005
beta1              = 0.0
beta2              = 0.9
#############################
# Make the Student networks and  optimizers
#############################
student_vae = SNNVAE().cuda(device)
student_classifier = My_Classifier_mnist_3d().cuda(device)
student_vae_encoder_optimizer = torch.optim.Adam(student_vae.fsencoder.parameters(),lr, (beta1, beta2))
student_vae_decoder_optimizer = torch.optim.Adam(student_vae.fsdecoder.parameters(),lr, (beta1, beta2))
student_classifier_optimizer = torch.optim.Adam(student_classifier.parameters(),lr,(beta1, beta2))
