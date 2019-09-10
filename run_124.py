import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import math
import models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import time

import os
ROOT = os.path.dirname(os.path.realpath(__file__))

METHOD = "DMPHN_1_2_4"

import iio

def read_tensor(path, tobatch=True, cpu=False):
    v = iio.read(path)
    v = torch.FloatTensor(v, device='cpu')
    if not cpu and torch.cuda.is_available():
        v = v.cuda()
    v = v.permute((2,0,1))
    if tobatch:
        v = torch.stack([v], dim=0)
    return v

def write_tensor(path, tensor):
    tensor = tensor.permute((0, 2, 3, 1)).squeeze()
    iio.write(path, tensor.cpu().detach().numpy())

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def deblur(input, output, normalization=1, GPU=0, load=f'{ROOT}/checkpoints/{METHOD}'):
    assert(type(input) == type(output))
    if type(input) not in (tuple, list):
        input = (input,)
        output = (output,)

    encoder_lv1 = models.Encoder().apply(weight_init).cuda(GPU)
    encoder_lv2 = models.Encoder().apply(weight_init).cuda(GPU)
    encoder_lv3 = models.Encoder().apply(weight_init).cuda(GPU)

    decoder_lv1 = models.Decoder().apply(weight_init).cuda(GPU)
    decoder_lv2 = models.Decoder().apply(weight_init).cuda(GPU)
    decoder_lv3 = models.Decoder().apply(weight_init).cuda(GPU)

    encoder_lv1.load_state_dict(torch.load(f'{load}/encoder_lv1.pkl'))
    encoder_lv2.load_state_dict(torch.load(f'{load}/encoder_lv2.pkl'))
    encoder_lv3.load_state_dict(torch.load(f'{load}/encoder_lv3.pkl'))

    decoder_lv1.load_state_dict(torch.load(f'{load}/decoder_lv1.pkl'))
    decoder_lv2.load_state_dict(torch.load(f'{load}/decoder_lv2.pkl'))
    decoder_lv3.load_state_dict(torch.load(f'{load}/decoder_lv3.pkl'))

    for input, output in zip(input, output):
        print(input, output)
        with torch.no_grad():
            images_lv1 = read_tensor(input) / normalization
            images_lv1 = Variable(images_lv1 - 0.5).cuda(GPU)
            H = images_lv1.size(2)
            W = images_lv1.size(3)

            images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
            images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
            images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
            images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
            images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
            images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]

            feature_lv3_1 = encoder_lv3(images_lv3_1)
            feature_lv3_2 = encoder_lv3(images_lv3_2)
            feature_lv3_3 = encoder_lv3(images_lv3_3)
            feature_lv3_4 = encoder_lv3(images_lv3_4)
            feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
            feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
            feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
            residual_lv3_top = decoder_lv3(feature_lv3_top)
            residual_lv3_bot = decoder_lv3(feature_lv3_bot)

            feature_lv2_1 = encoder_lv2(images_lv2_1 + residual_lv3_top)
            feature_lv2_2 = encoder_lv2(images_lv2_2 + residual_lv3_bot)
            feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
            residual_lv2 = decoder_lv2(feature_lv2)

            feature_lv1 = encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
            deblur_image = decoder_lv1(feature_lv1)

            write_tensor(output, (deblur_image+0.5) * normalization)

if __name__ == '__main__':
    import fire
    fire.Fire(deblur)

