import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.quantization import QuantStub, DeQuantStub

class Log(nn.Module):
    def __init__(self):
        super(Log, self).__init__()
             
    def forward(self, x):
        print(x.shape)
        return x
    
    
class ConvNormReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, use_bn=False):
        if use_bn:
            norm = nn.BatchNorm2d
        else:
            norm = nn.InstanceNorm2d
            
        super(ConvNormReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False),
            norm(out_planes, affine=True),
            # Replace with ReLU
            nn.ReLU(inplace=True)
        )
        
class ConvNorm(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, use_bn=False):
        if use_bn:
            norm = nn.BatchNorm2d
        else:
            norm = nn.InstanceNorm2d
            
        super(ConvNorm, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False),
            norm(out_planes)
        )

class ResidualBlock(nn.Module):
    def __init__(self, in_features, use_bn):
        super(ResidualBlock, self).__init__()
        
        if use_bn:
            norm = nn.BatchNorm2d
        else:
            norm = nn.InstanceNorm2d
            
        conv_block = [#  nn.ReflectionPad2d(1),
                       # nn.Conv2d(in_features, in_features, 3),
                       # norm(in_features),
                       # nn.ReLU(inplace=True),
                        ConvNormReLU(in_features, in_features, 3,  padding=1, use_bn=use_bn),
                       # nn.ReflectionPad2d(1),
                       # nn.Conv2d(in_features, in_features, 3),
                       # norm(in_features) 
                        ConvNorm(in_features, in_features, 3, padding=1, use_bn=use_bn)]

        self.conv_block = nn.Sequential(*conv_block)
        self.addition = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.addition.add(x, self.conv_block(x))
    

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, use_bn=False):
        super(Generator, self).__init__()
        if use_bn:
            norm = nn.BatchNorm2d
        else:
            norm = nn.InstanceNorm2d
            
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # Initial convolution block       
        model = [  # nn.ReflectionPad2d(3),
                    ConvNormReLU(input_nc, 64, 7, padding=3, use_bn=use_bn) ]
                   # nn.Conv2d(input_nc, 64, 7),
                   # norm(64),
                   # nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  ConvNormReLU(in_features, out_features, 3, stride=2, padding=1, use_bn=use_bn)]
                        #nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        #norm(out_features),
                        #nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features, use_bn)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [ 
                        nn.Upsample(scale_factor=2),
                      #  nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                     # #  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      #  norm(out_features),
                       # nn.ReLU(inplace=True)
                       ConvNormReLU(in_features, out_features, 3, stride=1, padding=1, use_bn=use_bn)]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [ # nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7, padding=3),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)
    
    def fuse(self):
        for m in self.model.modules():
            if type(m) == ConvNormReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == ConvNorm:
                torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
            

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0])
