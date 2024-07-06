import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ehbs import EHBSFeatureSelector
from models.gumble import FeatureSelectorGumble
from models.concrete_autoencoder import ConcreteEncoder


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batchnorm=True):
        super(Conv2DBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, eps=0.00001))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, eps=0.00001))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, input_channels,n_target_channels, n_classes,  band_selection=False, mask=None,batchnorm=True, enc_conv_kernel_size=3, enc_depth=4, n_filters=8, dropout=0.5, implicit_norm=False, inference=False, data_augmentation=False):
        super(UNet, self).__init__()
            #def __init__(self, n_channels,n_target_channels, n_classes, band_selection=False, mask=None, bilinear=False):

        self.band_selection = band_selection
        if band_selection:
            self.ehbs = ConcreteEncoder(
                input_dim=input_channels,output_dim=n_target_channels,device="cuda"
            )
            #self.ehbs = EHBSFeatureSelector(#input_dim,target_dim, sigma,
            #     input_dim=input_channels,target_dim=n_target_channels,sigma=0.5,device="cuda"
            #)

        self.mask = mask
        self.n_channels = n_target_channels
        self.n_classes = n_classes
        
        self.enc_depth = enc_depth
        self.inference = inference
        self.dropout = nn.Dropout(dropout)
        self.data_augmentation = DataAugmentation() if data_augmentation else None

        if implicit_norm:
            self.implicit_norm = nn.BatchNorm2d(self.n_channels, eps=0, affine=True, track_running_stats=False)
            self.implicit_norm.weight.data = torch.tensor([2.0])
            self.implicit_norm.bias.data = torch.tensor([-1.0])
        else:
            self.implicit_norm = None

        self.encoder = self.build_encoder(self.n_channels, n_filters, enc_conv_kernel_size, batchnorm)
        self.middle_conv = Conv2DBlock(n_filters * (2 ** (enc_depth - 1)), n_filters * (2 ** enc_depth), kernel_size=3, batchnorm=batchnorm)
        self.decoder = self.build_decoder(n_filters * (2 ** enc_depth), n_filters, enc_conv_kernel_size, batchnorm, n_classes)

    def build_encoder(self, in_channels, n_filters, kernel_size, batchnorm):
        layers = []
        for i in range(self.enc_depth):
            layers.append(Conv2DBlock(in_channels, n_filters, kernel_size, batchnorm))
            in_channels = n_filters
            n_filters *= 2
        return nn.ModuleList(layers)

    def build_decoder(self, in_channels, n_filters, kernel_size, batchnorm, n_classes):
        layers = []
        for i in range(self.enc_depth):
            layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
            in_channels //= 2
            layers.append(Conv2DBlock(in_channels * 2, in_channels, kernel_size, batchnorm))
        layers.append(nn.Conv2d(in_channels, n_classes, kernel_size=1))
        layers.append(nn.Softmax(dim=1))
        return nn.ModuleList(layers)

    def forward(self, x):
        if not self.training and self.mask is None:
            #print("self.training",self.training)
            cur_mask = torch.argmax(torch.from_numpy(self.ehbs.get_gates("raw")[0]), dim=1)
            x = x[:, cur_mask]
        #print(self.band_selection,self.mask)
        elif self.band_selection:
            x = self.ehbs(x)
        elif self.mask is not None:
            x = x[:, self.mask]
        if self.implicit_norm:
            x = self.implicit_norm(x)
        if self.data_augmentation:
            x = self.data_augmentation(x)

        encoder_outs = []
        for layer in self.encoder:
            x = layer(x)
            encoder_outs.append(x)
            x = F.max_pool2d(x, 2)

        x = self.middle_conv(x)

        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                x = torch.cat([x, encoder_outs[-(i // 2 + 1)]], dim=1)
            else:
                x = layer(x)

        if self.inference:
            return x[:, :self.n_classes, :, :]
        return x

# Example usage
#model=UNet(25, 5, 10, band_selection=False,mask=[1,3,11,13,20])
#model = UNet(input_channels=3, n_classes=2)
#input_tensor = torch.randn(1, 25, 256, 256)
#output = model(input_tensor)
#print(output.shape)
