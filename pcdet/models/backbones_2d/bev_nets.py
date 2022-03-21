import torch
import torch.nn as nn

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPSepConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, in_channels, 3, padding=dilation, dilation=dilation, bias=False, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        ]
        super(ASPPSepConv, self).__init__(*modules)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3, rate4 = atrous_rates
        modules.append(ASPPSepConv(in_channels, out_channels, rate1))
        modules.append(ASPPSepConv(in_channels, out_channels, rate2))
        modules.append(ASPPSepConv(in_channels, out_channels, rate3))
        modules.append(ASPPSepConv(in_channels, out_channels, rate4))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class UpHead8x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up_convs = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size= 2, stride= 2, padding= 0, bias=False),
            nn.BatchNorm2d(in_channels // 2, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(in_channels // 4, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(in_channels // 8, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.up_convs(x)

class DownHead8x(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.aspp = ASPP(in_channels, out_channels // 8, (2,3,4,5))
        self.downsample = nn.Sequential(
            self._make_conv_layer(out_channels // 8, out_channels // 4),
            self._make_conv_layer(out_channels // 4, out_channels // 2),
            self._make_conv_layer(out_channels // 2, out_channels)
        )

    def _make_conv_layer(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride= 2, padding=0),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        return conv_layer

    def forward(self, x):
        #x = self.aspp(x)
        x = self.downsample(x)
        return x