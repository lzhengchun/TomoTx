import torch
import torch.nn as nn
import torch.nn.functional as F

def model_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class unet_box(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.01), 
            torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.01),             
        )
    def forward(self, x):
        return self.double_conv(x)
    
class unet_bottleneck(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.01),        
        )
    def forward(self, x):
        return self.bn_conv(x)
    
class unet_up(torch.nn.Module):
    def __init__(self, ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.down_scale = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), )
        else:
            self.down_scale = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2), 
            torch.nn.LeakyReLU(negative_slope=0.01), )
            
    def forward(self, x):
        return self.down_scale(x)
            
class unet_down(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.maxpool = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), )
        
    def forward(self, x):
        return self.maxpool(x)
    
class unet(torch.nn.Module):
    def __init__(self, ich=1, och=1):
        super().__init__()
        self.in_box= torch.nn.Sequential(
            torch.nn.Conv2d(ich, 8, kernel_size=1, padding=0), 
            torch.nn.LeakyReLU(negative_slope=0.01),  )
        self.box1  = unet_box(8, 32)
        self.down1 = unet_down(32)

        self.box2  = unet_box(32, 64)
        self.down2 = unet_down(64)
        
        self.box3  = unet_box(64, 128)
        self.down3 = unet_down(128)
        
        self.bottleneck = unet_bottleneck(128, 128)
        
        self.up1   = unet_up(128)
        self.box4  = unet_box(256, 64)
        
        self.up2   = unet_up(64)
        self.box5  = unet_box(128, 32)
        
        self.up3   = unet_up(32)
        self.box6  = unet_box(64, 32)
        
        self.out_layer = torch.nn.Sequential(
            torch.nn.Conv2d(32, 16, kernel_size=1, padding=0), 
            torch.nn.LeakyReLU(negative_slope=0.01),  
            torch.nn.Conv2d(16, och, kernel_size=1, padding=0), )
        
    def forward(self, x):
        _in_conv1d = self.in_box(x)
        _box1_out  = self.box1(_in_conv1d)
        _down1_out = self.down1(_box1_out)
        
        _box2_out  = self.box2(_down1_out)
        _down2_out = self.down2(_box2_out)
        
        _box3_out  = self.box3(_down2_out)
        _down3_out = self.down3(_box3_out)
        
        _bottle_neck = self.bottleneck(_down3_out)
        
        _up1_out     = self.up1(_bottle_neck)
        up1_box3_cat = torch.cat((_box3_out, _up1_out), dim=1)
        
        _box4_out    = self.box4(up1_box3_cat)
        _up2_out     = self.up2(_box4_out)
        _up2_box2_cat= torch.cat((_box2_out, _up2_out), dim=1)
        
        _box5_out    = self.box5(_up2_box2_cat)
        _up3_out     = self.up3(_box5_out)
        _up3_box1_cat= torch.cat((_box1_out, _up3_out), dim=1)
        
        _box6_out    = self.box6(_up3_box1_cat)
        
        _output      = self.out_layer(_box6_out)
        return _output
