import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *

class DepthDecoder_learned_camera(nn.Module):
    def __init__(self, num_ch, scales=range(4), num_bins=32, num_out_chs=1, skips_connect=True):
        super(DepthDecoder_learned_camera, self).__init__()

        self.num_ch = num_ch
        self.scales = scales
        self.num_bins = num_bins
        self.num_out_chs = num_out_chs
        self.skips_connect = skips_connect

        self.upsample_mode = 'nearest'
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.conv = OrderedDict()
        self.feat_size = 512
        for i in range(4, -1, -1):
            # upconv_0
            if i == 4:
                num_ch_in = self.num_ch[-1]
            else:
                num_ch_in = self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.conv[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.skips_connect and i > 0:
                num_ch_in += self.num_ch[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.conv[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.conv[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_out_chs)

        self.conv[("relu1")] = nn.ReLU()
        self.conv[("relu2")] = nn.ReLU()
        self.conv[("relu3")] = nn.ReLU()
        self.conv[("relu4")] = nn.ReLU()
        self.conv[("relu5")] = nn.ReLU()
        self.conv[("sig_c")] = nn.Softmax(dim=1)

        self.conv[("conv1")] = nn.Conv2d(self.feat_size, self.feat_size, 3, 2, 1)
        self.conv[("conv2")] = nn.Conv2d(self.feat_size, self.feat_size, 3, 2, 1)
        self.conv[("conv3")] = nn.Conv2d(self.feat_size, self.feat_size, 3, 2, 1)
        self.conv[("linear1")] = nn.Linear(2*self.feat_size, int(self.feat_size/2))
        self.conv[("linear2")] = nn.Linear(int(self.feat_size/2), 2*self.num_bins+4)

        self.decoder = nn.ModuleList(list(self.conv.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_feat):
        self.outputs = {}

        # decoder
        z = input_feat[-1]
        for i in range(4, -1, -1):
            z = self.conv[("upconv", i, 0)](z)
            z = [upsample(z)]
            if self.skips_connect and i > 0:
                z += [input_feat[i - 1]]
            z = torch.cat(z, 1)
            z = self.conv[("upconv", i, 1)](z)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.conv[("dispconv", i)](z))

        zc = input_feat[-1]
        batch_size, in_ch, _, _ = zc.size()
        z1 = self.conv[("relu1")](self.conv[("conv1")](zc))
        z2 = self.conv[("relu2")](self.conv[("conv2")](z1))
        z3 = self.conv[("relu3")](self.conv[("conv3")](z2))
        z4 = self.conv[("relu4")](self.conv[("linear1")](z3.reshape(batch_size, -1)))
        z5 = self.conv[("linear2")](z4)
        d_alpha = self.conv[("relu5")](z5[:,0:self.num_bins])
        d_d = self.conv[("relu5")](z5[:,self.num_bins:2*self.num_bins])
        bin_width = d_d/(torch.sum(d_d, dim=1, keepdim=True) + 1e-7)

        camera_range = self.sigmoid(z5[:,2*self.num_bins:2*self.num_bins + 2])
        camera_offset = 2.0*(self.sigmoid(z5[:,2*self.num_bins + 2:2*self.num_bins + 4]) - 0.5)

        cam_lens_x = []
        for i in range(self.num_bins):
            lens_height = 0.0
            for j in range(0, i+1):
                lens_height += d_alpha[:, j:j+1]
            cam_lens_x.append(lens_height)
        cam_lens_c = torch.cat(cam_lens_x, dim=1)
        camera_param = cam_lens_c*bin_width/(torch.sum(cam_lens_c*bin_width, dim=1, keepdim=True) + 1e-7)

        return self.outputs, camera_param, bin_width, camera_range, camera_offset
