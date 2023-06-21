from lib.networks.UNet.UNet import UNet
import torch
import torch.nn as nn
import numpy as np
from sklearn.random_projection import GaussianRandomProjection

from .layers.netvlad import NetVLAD

class DescNet(nn.Module):
    def __init__(self, use_vlad = True, use_arcface=False, descriptor_size=1024, n_clusters=16, nchannels = 3, nclasses = 1, nangles = 18):
        super(DescNet, self).__init__()
        #self.backbone = SonarNet(nchannels, nclasses, nangles)
        self.backbone = UNet(nchannels, nclasses, False)
        self.use_arcface = use_arcface
        self.use_vlad = use_vlad
        if self.use_vlad:
            self.vlad = NetVLAD(dim=descriptor_size, num_clusters=n_clusters)
            print('conv descriptor size:',descriptor_size,'- netvlad_clusters:',n_clusters)
        else: 
            if self.use_arcface:
                print('conv descriptor size:',descriptor_size)
                self.fc = nn.Linear(512*25*32, descriptor_size, bias = False)
                self.features = nn.BatchNorm1d(descriptor_size, eps=1e-5)
                nn.init.constant_(self.features.weight,1.0)
                self.features.weight.requires_grad = False
                self.head = nn.Sequential(
                    self.fc,
                    self.features
                )
            else:
                dummy = np.zeros((1, 512*25*32))
                transformer = GaussianRandomProjection(n_components=descriptor_size, random_state = 42)
                transformer.fit(dummy)
                self.proj_mat = torch.from_numpy(transformer.components_.astype(np.float32)).cuda()
                self.proj_mat.requires_grad = False


    def forward(self, input):
        xfc = self.get_feature_map(input)
        output = self.apply_head(xfc)
        return output

    def get_feature_map(self, input):
        x1, x2, x3, x4, x5 = self.backbone.get_feature_map(input)
        if x5.shape[2] == 12:
            x5 = nn.functional.interpolate(x5, (13,x5.shape[3]), mode='bilinear', align_corners=False)
        return x4

    def apply_head(self, input):
        if self.use_vlad:
            output = self.vlad(input)
        else:
            if self.use_arcface:
                flat_input = torch.flatten(input, start_dim=1) 
                output = self.head(flat_input)
            else:
                flat_input = torch.flatten(input, start_dim=1) # bxfeat
                flat_input = flat_input.permute(1,0) # featxb
                output = torch.matmul(self.proj_mat, flat_input) # dimxb
                output = output.permute(1,0) # bxdim
        output = torch.nn.functional.normalize(output, p=2, dim = 1)
        return output

def get_desc_net(use_vlad, use_arcface, descriptor_size, n_clusters):
    return DescNet(use_vlad = use_vlad, use_arcface=use_arcface, descriptor_size=descriptor_size, n_clusters=n_clusters)
