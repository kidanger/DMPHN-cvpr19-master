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

METHOD = "SDNet3"

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

def deblur(input, output, normalization=1, GPU=0):
    assert(type(input) == type(output))
    if type(input) not in (tuple, list):
        input = (input,)
        output = (output,)

    encoder = {}
    decoder = {}
    for s in ['s1', 's2', 's3']:
        encoder[s] = {}
        decoder[s] = {}
        for lv in ['lv1', 'lv2', 'lv3']:
            encoder[s][lv] = models.Encoder()
            decoder[s][lv] = models.Decoder()
            encoder[s][lv].apply(weight_init).cuda(GPU)
            decoder[s][lv].apply(weight_init).cuda(GPU)
            encoder[s][lv].load_state_dict(torch.load(f'{ROOT}/checkpoints/{METHOD}/encoder_{s}_{lv}.pkl'))
            decoder[s][lv].load_state_dict(torch.load(f'{ROOT}/checkpoints/{METHOD}/decoder_{s}_{lv}.pkl'))

    for input, output in zip(input, output):
        print(input, output)
        with torch.no_grad():
            images = {}
            feature = {}
            residual = {}
            for s in ['s1', 's2', 's3']:
                feature[s] = {}
                residual[s] = {}

            images['lv1'] = read_tensor(input) / normalization
            images['lv1'] = Variable(images['lv1'] - 0.5).cuda(GPU)

            H = images['lv1'].size(2)
            W = images['lv1'].size(3)

            images['lv2_1'] = images['lv1'][:,:,0:int(H/2),:]
            images['lv2_2'] = images['lv1'][:,:,int(H/2):H,:]
            images['lv3_1'] = images['lv2_1'][:,:,:,0:int(W/2)]
            images['lv3_2'] = images['lv2_1'][:,:,:,int(W/2):W]
            images['lv3_3'] = images['lv2_2'][:,:,:,0:int(W/2)]
            images['lv3_4'] = images['lv2_2'][:,:,:,int(W/2):W]

            s = 's1'
            feature[s]['lv3_1'] = encoder[s]['lv3'](images['lv3_1'])
            feature[s]['lv3_2'] = encoder[s]['lv3'](images['lv3_2'])
            feature[s]['lv3_3'] = encoder[s]['lv3'](images['lv3_3'])
            feature[s]['lv3_4'] = encoder[s]['lv3'](images['lv3_4'])
            feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3)
            feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3)
            residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
            residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

            feature[s]['lv2_1'] = encoder[s]['lv2'](images['lv2_1'] + residual[s]['lv3_top']) + feature[s]['lv3_top']
            feature[s]['lv2_2'] = encoder[s]['lv2'](images['lv2_2'] + residual[s]['lv3_bot']) + feature[s]['lv3_bot']
            feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
            residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2'])

            feature[s]['lv1'] = encoder[s]['lv1'](images['lv1'] + residual[s]['lv2']) + feature[s]['lv2']
            residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])

            s = 's2'
            ps = 's1'
            feature[s]['lv3_1'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),0:int(W/2)])
            feature[s]['lv3_2'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),int(W/2):W])
            feature[s]['lv3_3'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,0:int(W/2)])
            feature[s]['lv3_4'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,int(W/2):W])
            feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3) + feature[ps]['lv3_top']
            feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3) + feature[ps]['lv3_bot']
            residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
            residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

            feature[s]['lv2_1'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,0:int(H/2),:] + residual[s]['lv3_top']) + feature[s]['lv3_top'] + feature[ps]['lv2_1']
            feature[s]['lv2_2'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,int(H/2):H,:] + residual[s]['lv3_bot']) + feature[s]['lv3_bot'] + feature[ps]['lv2_2']
            feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
            residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2']) + residual['s1']['lv1']

            feature[s]['lv1'] = encoder[s]['lv1'](residual[ps]['lv1'] + residual[s]['lv2']) + feature[s]['lv2'] + feature[ps]['lv1']
            residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])

            s = 's3'
            ps = 's2'
            feature[s]['lv3_1'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),0:int(W/2)])
            feature[s]['lv3_2'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,0:int(H/2),int(W/2):W])
            feature[s]['lv3_3'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,0:int(W/2)])
            feature[s]['lv3_4'] = encoder[s]['lv3'](residual[ps]['lv1'][:,:,int(H/2):H,int(W/2):W])
            feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3) + feature[ps]['lv3_top']
            feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3) + feature[ps]['lv3_bot']
            residual[s]['lv3_top'] = decoder[s]['lv3'](feature[s]['lv3_top'])
            residual[s]['lv3_bot'] = decoder[s]['lv3'](feature[s]['lv3_bot'])

            feature[s]['lv2_1'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,0:int(H/2),:] + residual[s]['lv3_top']) + feature[s]['lv3_top'] + feature[ps]['lv2_1']
            feature[s]['lv2_2'] = encoder[s]['lv2'](residual[ps]['lv1'][:,:,int(H/2):H,:] + residual[s]['lv3_bot']) + feature[s]['lv3_bot'] + feature[ps]['lv2_2']
            feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)
            residual[s]['lv2'] = decoder[s]['lv2'](feature[s]['lv2']) + residual['s1']['lv1']

            feature[s]['lv1'] = encoder[s]['lv1'](residual[ps]['lv1'] + residual[s]['lv2']) + feature[s]['lv2'] + feature[ps]['lv1']
            residual[s]['lv1'] = decoder[s]['lv1'](feature[s]['lv1'])

            deblurred_image = residual[s]['lv1']

            write_tensor(output, (deblurred_image+0.5) * normalization)

if __name__ == '__main__':
    import fire
    fire.Fire(deblur)

