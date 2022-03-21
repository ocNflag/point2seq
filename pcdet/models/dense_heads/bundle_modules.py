import torch.nn as nn
from pcdet.models.dense_heads.utils import Sequential, FeatureAdaption


class DCNBundleHead(nn.Module):
    def __init__(self, in_channels, num_cls, heads, head_conv=64, final_kernel=1, bn=False, init_bias=-2.19, **kwargs):
        super(DCNBundleHead, self).__init__(**kwargs)

        # feature adaptation with dcn
        # use separate features for classification / regression
        self.feature_adapt_cls = FeatureAdaption(in_channels, in_channels, kernel_size=3, deformable_groups=4)
        self.feature_adapt_reg = FeatureAdaption(in_channels, in_channels, kernel_size=3, deformable_groups=4)

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(head_conv, num_cls, kernel_size=3, stride=1, padding=1, bias=True)
        self.center_head.bias.data.fill_(init_bias)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(head_conv, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.corner_head.bias.data.fill_(init_bias)

        self.fg_conv = Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_cls, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.fg_head = nn.Conv2d(head_conv, num_cls, kernel_size=3, stride=1, padding=1, bias=True)
        self.fg_head.bias.data.fill_(init_bias)

        # other regression target
        self.task_head = NaiveSepHead(in_channels, heads, head_ch=head_conv, bn=bn, final_kernel=final_kernel)

    def forward(self, x):
        map_feat = self.feature_adapt_cls(x)
        reg_feat = self.feature_adapt_reg(x)

        ret = self.task_head(reg_feat)

        center_feat = self.center_conv(map_feat)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(map_feat)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(map_feat)
        fg_map = self.fg_head(fg_feat)

        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map

        return ret
