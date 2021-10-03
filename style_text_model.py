import torch
import torch.nn.functional as F
from torch import nn


class Residual(nn.Module):
    def __init__(self, in_dim):
        super(Residual, self).__init__()
        temp_channels = in_dim // 4
        self.conv = nn.Sequential(nn.Conv2d(in_dim, temp_channels, kernel_size=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(temp_channels, temp_channels,
                                            kernel_size=3, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2),
                                  nn.Conv2d(temp_channels, in_dim, kernel_size=1))
        self.bn = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        out = self.conv(x) + x
        return F.leaky_relu(self.bn(out), negative_slope=0.2)


class ResNet(nn.Module):
    def __init__(self, in_dim):
        super(ResNet, self).__init__()
        self.layer = nn.Sequential(Residual(in_dim),
                                   Residual(in_dim),
                                   Residual(in_dim),
                                   Residual(in_dim))

    def forward(self, x):
        return self.layer(x)


def conv_bn_relu(in_channels, out_channels):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(negative_slope=0.2))
    return blk


class EncoderNet(nn.Module):
    def __init__(self, in_dim):
        super(EncoderNet, self).__init__()
        layer1, layer2, layer3, layer4 = [], [], [], []

        layer1.append(conv_bn_relu(in_dim, 32))
        layer1.append(conv_bn_relu(32, 32))

        layer2.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        layer2.append(nn.LeakyReLU(negative_slope=0.2))
        layer2.append(conv_bn_relu(64, 64))
        layer2.append(conv_bn_relu(64, 64))

        layer3.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        layer3.append(nn.LeakyReLU(negative_slope=0.2))
        layer3.append(conv_bn_relu(128, 128))
        layer3.append(conv_bn_relu(128, 128))

        layer4.append(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        layer4.append(nn.LeakyReLU(negative_slope=0.2))
        layer4.append(conv_bn_relu(256, 256))
        layer4.append(conv_bn_relu(256, 256))

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)

    def forward(self, x, get_feature_map=False):
        out = self.l1(x)
        out = self.l2(out)
        f1 = out
        out = self.l3(out)
        f2 = out
        out = self.l4(out)
        if get_feature_map:
            return out, [f2, f1]
        else:
            return out


class DecoderNet(nn.Module):
    def __init__(self, in_dim, feature_map_channels=None):
        super(DecoderNet, self).__init__()

        f1, f2, f3 = 0, 0, 0
        if feature_map_channels:
            f1, f2, f3 = feature_map_channels

        cat_channels = in_dim + f1
        self.conv1 = nn.Sequential(conv_bn_relu(cat_channels, 256),
                                   conv_bn_relu(256, 256))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.LeakyReLU(negative_slope=0.2))

        cat_channels = 128 + f2
        self.conv2 = nn.Sequential(conv_bn_relu(cat_channels, 128),
                                   conv_bn_relu(128, 128))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.LeakyReLU(negative_slope=0.2))

        cat_channels = 64 + f3
        self.conv3 = nn.Sequential(conv_bn_relu(cat_channels, 64),
                                   conv_bn_relu(64, 64))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(conv_bn_relu(32, 32),
                                   conv_bn_relu(32, 32))

    def forward(self, x, fuse=None, get_feature_map=False):
        if fuse and fuse[0] is not None:
            x = torch.cat([x, fuse[0]], dim=1)
        out = self.conv1(x)
        f1 = out

        out = self.deconv1(out)
        if fuse and fuse[1] is not None:
            out = torch.cat([out, fuse[1]], dim=1)
        out = self.conv2(out)
        f2 = out

        out = self.deconv2(out)
        if fuse and fuse[2] is not None:
            out = torch.cat([out, fuse[2]], dim=1)
        out = self.conv3(out)
        f3 = out

        out = self.deconv3(out)
        out = self.conv4(out)

        if get_feature_map:
            return out, [f1, f2, f3]
        else:
            return out


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width*height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out


class TextConversionNet(nn.Module):
    def __init__(self, in_dim=3):
        super(TextConversionNet, self).__init__()
        self.t_encoder = nn.Sequential(EncoderNet(in_dim),
                                       ResNet(256))
        self.s_encoder = nn.Sequential(EncoderNet(in_dim),
                                       ResNet(256))

        self.self_attn = Self_Attn(2 * 256)

        self.decoder = DecoderNet(2 * 256)

        self.last = nn.Sequential(conv_bn_relu(32, 32),
                                  nn.Conv2d(32, 3, kernel_size=3, padding=1),
                                  nn.Tanh())

    def forward(self, x_t, x_s):
        out_t = self.t_encoder(x_t)
        out_s = self.s_encoder(x_s)

        out = torch.cat([out_t, out_s], dim=1)

        out = self.self_attn(out)

        out = self.decoder(out)

        return self.last(out)


class BackgroundInpaintingNet(nn.Module):
    def __init__(self, in_dim=3):
        super(BackgroundInpaintingNet, self).__init__()
        self.encoder = EncoderNet(in_dim)
        self.resnet = ResNet(256)
        self.decoder = DecoderNet(256, [0, 128, 64])
        self.last = nn.Sequential(nn.Conv2d(32, 3, kernel_size=3, padding=1),
                                  nn.Tanh())

    def forward(self, x):
        out, f_encoder = self.encoder(x, get_feature_map=True)
        out = self.resnet(out)
        # out = self.dilation_net(out)
        out, fuse = self.decoder(out, [None] + f_encoder, get_feature_map=True)
        return self.last(out), fuse


class FusionNet(nn.Module):
    def __init__(self, in_dim=3):
        super(FusionNet, self).__init__()
        self.encoder = EncoderNet(in_dim)
        self.resnet = ResNet(256)
        self.decoder = DecoderNet(256, [256, 128, 64])
        self.last = nn.Sequential(nn.Conv2d(32, 3, kernel_size=3, padding=1),
                                  nn.Tanh())

    def forward(self, x, fuse):
        out = self.encoder(x)
        out = self.resnet(out)
        out = self.decoder(out, fuse)
        return self.last(out)


class Generator(nn.Module):
    def __init__(self, in_dim=3):
        super(Generator, self).__init__()
        self.text_conversion_net = TextConversionNet(in_dim)
        self.background_inpainting_net = BackgroundInpaintingNet(in_dim)
        self.fusion_net = FusionNet(in_dim)

    def forward(self, inputs):
        i_t, i_s = inputs
        o_t = self.text_conversion_net(i_t, i_s)
        o_b, fuse = self.background_inpainting_net(i_s)
        o_f = self.fusion_net(o_t, fuse)
        return o_t, o_b, o_f
