from .resnet import resnet18
import torch
import torch.nn as nn
from .layers.netvlad import NetVLAD
import numpy as np
from sklearn.random_projection import GaussianRandomProjection

# from https://github.com/artste/fastai-samples/blob/master/kaggle/lesson2-protein-human-protein-atlas-384-resnet50_data_block.ipynb
def make_batches(x,bs):
    '''
    Sample make_batches(11,3) = [3,3,3,2]
    '''
    if(x<=bs):
        return [min(x,bs)]
    else:
        return [bs] + make_batches(x-bs,bs)

# from https://github.com/artste/fastai-samples/blob/master/kaggle/lesson2-protein-human-protein-atlas-384-resnet50_data_block.ipynb
def create_new_weights(original_weights,nChannels):
    dst = torch.zeros(64,nChannels,3,3)
    #Repeat original weights up to fill dimension
    start=0
    for i in make_batches(nChannels,3):
        #print('dst',start,start+i, ' = src',0,i)
        dst[:,start:start+i,:,:]=original_weights[:,:i,:,:]
        start = start+i
    return dst


# from https://github.com/artste/fastai-samples/blob/master/kaggle/lesson2-protein-human-protein-atlas-384-resnet50_data_block.ipynb
def adapt_first_layer(src_model, nChannels):
    '''
    Change first layer of network to accomodate new channels
    '''

    # TODO: check if it worked

    # save original
    original_weights = src_model.weight.clone()

    # create new repeating existent weights channelwise
    new_weights = create_new_weights(original_weights,nChannels)

    # create new layes
    new_layer = nn.Conv2d(nChannels,64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    new_layer.weight = nn.Parameter(new_weights)

    return new_layer

class Upscaler(nn.Module):
    def __init__(self, inplanes, outplanes,scale=2):
        super(Upscaler, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes,3,1,1, bias = False),
            nn.BatchNorm2d(outplanes),
            nn.LeakyReLU(0.1,True)
        )
        if scale:
            self.upsampler = nn.UpsamplingBilinear2d(scale_factor=scale) # 50,44 
        else:
            self.upsampler = nn.Identity()
    
    def forward(self,input):
        res = self.conv(input)
        return self.upsampler(res)

class SpatialMaxPooling(nn.Module):
    def __init__(self, precision, imsize, desc_transform):
        super(SpatialMaxPooling, self).__init__()
        self.n = 120//precision
        self.s = imsize[1] - self.n*precision
        self.precision = precision
        self.pooler = nn.MaxPool2d((imsize[0], self.s))
        self.desc_transform = desc_transform

    def forward(self,input):
        '''
        input: [b,c,h,w], s+n = w
        '''

        start_index = self.s//2
        half_width = self.s//2
        
        spatial_descriptors = []
        for i in range(self.n):
            curr_start = start_index + i*self.precision
#            desc = self.pooler(input[:,:,:,curr_start-half_width:curr_start+half_width]) # [b,c,1]
            desc = self.desc_transform(input[:,:,:,curr_start-half_width:curr_start+half_width]).unsqueeze(2) # [b,c,1]
#            desc = self.desc_transform(desc).unsqueeze(2)
            spatial_descriptors.append(desc)
        return torch.cat(spatial_descriptors, dim=2)

class DescNet(nn.Module):
    def __init__(self, use_vlad = True, use_arcface=False, descriptor_size=1024, n_clusters=16):
        super(DescNet, self).__init__()
        self.use_vlad = use_vlad
        self.use_arcface = use_arcface
        self.backbone = resnet18(fully_conv = True,
                            pretrained = False,
                            output_stride=32,
                            remove_avg_pool_layer=True)
        self.backbone.conv1 = adapt_first_layer(self.backbone.conv1, 1)
        
        self.backbone.fc = nn.Sequential(
            nn.Conv2d(self.backbone.inplanes, descriptor_size, 3, 1, 1, bias=False),
            nn.BatchNorm2d(descriptor_size),
            nn.ReLU(True))
            
        #self.backbone.conv1 = adapt_first_layer(self.backbone.conv1, 1)
        if self.use_vlad:
            self.vlad = NetVLAD(dim=descriptor_size, num_clusters=n_clusters)
            print('conv features:',descriptor_size,'- netvlad_clusters:',n_clusters)
        elif self.use_arcface:
            print('conv descriptor size:',descriptor_size)
            self.fc = nn.Linear(512*25*32, descriptor_size, bias = False)
            self.features = nn.BatchNorm1d(descriptor_size, eps=1e-5)
            nn.init.constant_(self.features.weight,1.0)
            self.features.weight.requires_grad = False
            self.head = nn.Sequential(
                self.fc,
                self.features
            )
            self.head = NetVLAD(dim=descriptor_size, num_clusters=n_clusters, circular=False)
        else:
            dummy = np.zeros((1, self.backbone.inplanes*25*32))
            transformer = GaussianRandomProjection(n_components=descriptor_size, random_state = 42)
            transformer.fit(dummy)
            self.proj_mat = torch.from_numpy(transformer.components_.astype(np.float32)).cuda()
            self.proj_mat.requires_grad = False

    def forward(self, input):
        xfc = self.get_feature_map(input)
        output = self.apply_head(xfc)
        return output

    def get_feature_map(self, input):
        x2s, x4s, x8s, x16s, x32s = self.backbone(input)
        return x32s

    def apply_head(self, input):
        if self.use_arcface:
            flat_input = torch.flatten(input, start_dim=1)
            output = self.head(flat_input)
        else:
            if self.use_vlad:
                input = self.backbone.fc(input)
                output = self.vlad(input)
            else:
                flat_input = torch.flatten(input, start_dim=1) # bxfeat
                flat_input = flat_input.permute(1,0) # featxb
                output = torch.matmul(self.proj_mat, flat_input) # dimxb
                output = output.permute(1,0) # bxdim
        output = torch.nn.functional.normalize(output, p=2, dim = 1)
        return output

def get_desc_net(use_vlad, use_arcface, descriptor_size, n_clusters):
    return DescNet(use_vlad = use_vlad, use_arcface=use_arcface, descriptor_size=descriptor_size, n_clusters=n_clusters)
