# encoding: utf-8

import functools

import torch
import torch.nn as nn
from models.module import Harm2d
import torch.nn.functional as F
import math
import numpy as np


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
def harm3x3(in_planes, out_planes, stride=1, level=None):
    """3x3 harmonic convolution with padding"""
    return Harm2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                  bias=False, use_bn=False, level=level)


def get_feature_cood(frequency_num):
    array = 2 * ((np.arange(frequency_num) * 1.0) / (frequency_num - 1)) - 1
    #array = np.random.random(frequency_num)
    return torch.FloatTensor(np.float32(array))  # .view(1, -1, 1, 1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=32):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mip = max(8, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, mip, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mip, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # return x * y.expand_as(x)
        return y


class attention(nn.Module):
    def __init__(self, channel, reduction=32):
        super(attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mip = max(8, channel // reduction)

        # nn.Linear(channel, mip, bias=False),
        # nn.ReLU(inplace=True),
        # nn.Linear(mip, channel, bias=False),
        # nn.Sigmoid()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        inp = channel
        oup = channel

        # mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_c = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        '''
        n, c, h, w = x.size()
        #print('---x.shape', x.shape)
        x_h = self.pool_h(x)
        #print('x_h.shape',x_h.shape)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        #print('x_w.shape',x_w.shape)
        x_c = self.avg_pool(x)

        y = torch.cat([x_h, x_w, x_c], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w, x_c = torch.split(y, [h, w, 1], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_c = self.conv_c(x_c).sigmoid()
        # return x * y.expand_as(x)
        return a_h * a_w * a_c


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        # """suppose a convolutional layer with g groups whose output has
        # g x n channels; we first reshape the output channel dimension
        # into (g, n)"""
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        # """transposing and then flattening it back as the input of next layer."""
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x


'''Hnet = UnetGenerator_C(input_nc=opt.channel_secret * opt.num_secret, output_nc=opt.channel_cover * opt.num_cover,
                             num_downs=num_downs, norm_layer=norm_layer, output_function=nn.Tanh)'''


class UnetGenerator_C(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=None, use_dropout=False, output_function=nn.Sigmoid):
        super(UnetGenerator_C, self).__init__()
        self.groups = 1
        self.factor = 10 / 255
        # self.bias = False
        self.tanh = nn.Tanh
        '''if self.tanh:

        else:
            self.factor = 1.0'''
        nf = 9
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(256)
        self.bn4 = norm_layer(512)
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU()
        self.convtran5 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnt5 = norm_layer(512)
        self.convtran4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnt4 = norm_layer(256)
        self.convtran3 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnt3 = norm_layer(128)
        self.convtran2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnt2 = norm_layer(64)
        self.convtran1 = nn.ConvTranspose2d(128, output_nc, kernel_size=4, stride=2, padding=1, bias=False)

        self.atten1 = attention(512)
        self.atten2 = attention(256)
        self.atten3 = attention(128)
        self.atten4 = attention(64)
        self.atten5 = attention(output_nc)
        self.channel_shuffle1 = ChannelShuffle(512)
        self.channel_shuffle2 = ChannelShuffle(256)
        self.channel_shuffle3 = ChannelShuffle(128)
        self.channel_shuffle4 = ChannelShuffle(64)
        self.channel_shuffle5 = ChannelShuffle(output_nc)
        self.dctconv1 = harm3x3(512, 512)
        self.dctconv2 = harm3x3(256, 256)
        self.dctconv3 = harm3x3(128, 128)
        self.dctconv4 = harm3x3(64, 64)
        self.dctconv5 = harm3x3(output_nc, output_nc)

        self.f_atten = get_feature_cood(nf)
        self.f_atten_1 = nn.Parameter(
            self.f_atten.repeat(512).view(1, -1, 1, 1), requires_grad=True)
        self.f_atten_2 = nn.Parameter(
            self.f_atten.repeat(256).view(1, -1, 1, 1), requires_grad=True)
        self.f_atten_3 = nn.Parameter(
            self.f_atten.repeat(128).view(1, -1, 1, 1), requires_grad=True)
        self.f_atten_4 = nn.Parameter(
            self.f_atten.repeat(64).view(1, -1, 1, 1), requires_grad=True)
        self.f_atten_5 = nn.Parameter(
            self.f_atten.repeat(output_nc).view(1, -1, 1, 1), requires_grad=True)
        self.weight5 = nn.Parameter(
            nn.init.kaiming_normal_(torch.Tensor(512, 512 * nf, 1, 1), mode='fan_out',
                                    nonlinearity='relu'))
        self.weight4 = nn.Parameter(
            nn.init.kaiming_normal_(torch.Tensor(256, 256 * nf, 1, 1), mode='fan_out',
                                    nonlinearity='relu'))
        self.weight3 = nn.Parameter(
            nn.init.kaiming_normal_(torch.Tensor(128, 128 * nf, 1, 1), mode='fan_out',
                                    nonlinearity='relu'))
        self.weight2 = nn.Parameter(
            nn.init.kaiming_normal_(torch.Tensor(64, 64 * nf, 1, 1), mode='fan_out',
                                    nonlinearity='relu'))
        self.weight1 = nn.Parameter(
            nn.init.kaiming_normal_(torch.Tensor(output_nc, output_nc * nf, 1, 1), mode='fan_out',
                                    nonlinearity='relu'))

        self.drop = nn.Dropout(0.5)
        nhf = 64
        self.conv1_r = nn.Conv2d(input_nc, nhf, 3, 1, 1)
        self.conv2_r = nn.Conv2d(nhf, nhf * 2, 3, 1, 1)
        # self.conv3_r = nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1)
        # self.conv4_r = nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1)
        self.conv5_r = nn.Conv2d(nhf * 2, nhf, 3, 1, 1)
        self.conv6_r = nn.Conv2d(nhf, output_nc, 3, 1, 1)
        self.norm1_r = norm_layer(nhf)
        self.norm2_r = norm_layer(nhf * 2)
        self.norm3_r = norm_layer(nhf * 4)
        self.norm4_r = norm_layer(nhf * 2)
        self.norm5_r = norm_layer(nhf)
        # self.output = output_function()
        self.relu_r = nn.ReLU(True)
        self.output = nn.Sigmoid()

    def forward(self, input, out_s_dct_1, out_s_dct_2, out_s_dct_3, out_s_dct_4, out_s_dct_5):
        out1 = self.conv1(input)
        out2 = self.bn2(self.conv2(self.leakyrelu(out1)))
        out3 = self.bn3(self.conv3(self.leakyrelu(out2)))
        out4 = self.bn4(self.conv4(self.leakyrelu(out3)))
        out5 = self.conv5(self.leakyrelu(out4))
        out_5 = self.bnt5(self.convtran5(self.relu(out5)))
        #print('out_5.shape',out_5.shape)
        out_dct_5 = self.dctconv1(out_5)
        # out_5_cs = self.atten1_CAM(out_5) + self.atten1_PAM(out_5)
        out_dct_5_cs = self.atten1(out_5)
        #print('attention.size', out_dct_5_cs.shape)
        out_dct_5_cs = self.channel_shuffle1(out_dct_5_cs.repeat(1, 9, 1, 1).expand_as(out_dct_5))  # *out_dct_1
        out_dct_5_f = self.f_atten_1.expand_as(out_dct_5).to(input.device)  # * out_5
        #print('out_dct_5_f', out_dct_5_f.shape)
        #print('self.f_atten_1', self.f_atten_1.shape)
        out_dct_5 = (out_dct_5_cs + out_dct_5_f) * out_dct_5
        out_dct_5 = out_dct_5 + out_s_dct_1
        out_5 = F.conv2d(out_dct_5, self.weight5, padding=0, groups=self.groups)
        out_5 = torch.cat([out4, out_5], 1)

        out_4 = self.bnt4(self.convtran4(self.relu(out_5)))
        out_4 = self.drop(out_4)
        out_dct_4 = self.dctconv2(out_4)
        # out_4_cs = self.atten2_CAM(out_4) + self.atten2_PAM(out_4)
        out_dct_4_cs = self.atten2(out_4)
        out_dct_4_cs = self.channel_shuffle2(out_dct_4_cs.repeat(1, 9, 1, 1).expand_as(out_dct_4))
        out_dct_4_f = self.f_atten_2.expand_as(out_dct_4).to(input.device)  # * out_4
        out_dct_4 = (out_dct_4_cs + out_dct_4_f) * out_dct_4
        out_dct_4 = out_dct_4 + out_s_dct_2
        out_4 = F.conv2d(out_dct_4, self.weight4, padding=0, groups=self.groups)
        out_4 = torch.cat([out3, out_4], 1)

        out_3 = self.bnt3(self.convtran3(self.relu(out_4)))
        out_dct_3 = self.dctconv3(out_3)
        # out_3_cs = self.atten3_CAM(out_3) + self.atten3_PAM(out_3)
        out_dct_3_cs = self.atten3(out_3)
        out_dct_3_cs = self.channel_shuffle3(out_dct_3_cs.repeat(1, 9, 1, 1).expand_as(out_dct_3))
        out_dct_3_f = self.f_atten_3.expand_as(out_dct_3).to(input.device)  # * out_3
        out_dct_3 = (out_dct_3_cs + out_dct_3_f) * out_dct_3
        out_dct_3 = out_dct_3 + out_s_dct_3
        out_3 = F.conv2d(out_dct_3, self.weight3, padding=0, groups=self.groups)
        out_3 = torch.cat([out2, out_3], 1)

        out_2 = self.bnt2(self.convtran2(self.relu(out_3)))
        out_dct_2 = self.dctconv4(out_2)
        # out_2_cs = self.atten4_CAM(out_2) + self.atten4_PAM(out_2)
        out_dct_2_cs = self.atten4(out_2)
        out_dct_2_cs = self.channel_shuffle4(out_dct_2_cs.repeat(1, 9, 1, 1).expand_as(out_dct_2))
        out_dct_2_f = self.f_atten_4.expand_as(out_dct_2).to(input.device)  # * out_2
        out_dct_2 = (out_dct_2_cs + out_dct_2_f) * out_dct_2
        out_2 = out_dct_2 + out_s_dct_4
        out_2 = F.conv2d(out_2, self.weight2, padding=0, groups=self.groups)
        out_2 = torch.cat([out1, out_2], 1)

        out_1 = self.convtran1(self.relu(out_2))
        out_dct_1 = self.dctconv5(out_1)
        # out_1_cs = self.atten5_CAM(out_1) + self.atten5_PAM(out_1)
        out_dct_1_cs = self.atten5(out_1)
        out_dct_1_cs = self.channel_shuffle5(out_dct_1_cs.repeat(1, 9, 1, 1).expand_as(out_dct_1))
        out_dct_1_f = self.f_atten_5.expand_as(out_dct_1).to(input.device)  # * out_1
        out_dct_1 = (out_dct_1_cs + out_dct_1_f) * out_dct_1
        out_dct_1 = out_dct_1 + out_s_dct_5
        out = F.conv2d(out_dct_1, self.weight1, padding=0, groups=self.groups)

        x = self.relu_r(self.norm1_r(self.conv1_r(out)))
        x = self.relu_r(self.norm2_r(self.conv2_r(x)))
        # x = self.relu_r(self.norm3_r(self.conv3_r(x)))
        # x = self.relu_r(self.norm4_r(self.conv4_r(x)))
        x = self.relu_r(self.norm5_r(self.conv5_r(x)))
        out = self.output(self.conv6_r(x))
        # out = self.factor * out

        # out = torch.tanh(out)
        # out = torch.cat([input, out], 1)

        return out


