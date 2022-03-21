import torch
from torch import nn
import torch.nn.functional as F

from pcdet.models.dense_heads.fusion_modules import DCNFusionV3
from pcdet.models.dense_heads.utils import Sequential, kaiming_init


class AttnSepHead(nn.Module):
    def __init__(self, base_ch, feat_ch, heads, key_ch, value_ch, init_bias=-2.19, **kwargs):
        super(AttnSepHead, self).__init__(**kwargs)

        self.heads = heads
        self.init_bias = init_bias

        self.base_queries_modules = nn.ModuleDict()
        self.center_key_modules = nn.ModuleDict()
        self.center_value_modules = nn.ModuleDict()
        self.corner_key_modules = nn.ModuleDict()
        self.corner_value_modules = nn.ModuleDict()
        self.fg_key_modules = nn.ModuleDict()
        self.fg_value_modules = nn.ModuleDict()
        self.predict_modules = nn.ModuleDict()

        for head in self.heads:
            out_ch, *_ = self.heads[head]
            self.base_queries_modules[head] = Sequential(
                nn.Conv2d(base_ch, key_ch, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(key_ch),
                nn.ReLU(),
            )
            self.center_key_modules[head] = Sequential(
                nn.Conv2d(feat_ch, key_ch, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(key_ch),
                nn.ReLU(),
            )
            self.center_value_modules[head] = Sequential(
                nn.Conv2d(feat_ch, value_ch, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(value_ch),
                nn.ReLU(),
            )
            self.corner_key_modules[head] = Sequential(
                nn.Conv2d(feat_ch, key_ch, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(key_ch),
                nn.ReLU(),
            )
            self.corner_value_modules[head] = Sequential(
                nn.Conv2d(feat_ch, value_ch, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(value_ch),
                nn.ReLU(),
            )
            self.fg_key_modules[head] = Sequential(
                nn.Conv2d(feat_ch, key_ch, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(key_ch),
                nn.ReLU(),
            )
            self.fg_value_modules[head] = Sequential(
                nn.Conv2d(feat_ch, value_ch, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(value_ch),
                nn.ReLU(),
            )
            self.predict_modules[head] = Sequential(
                nn.Conv2d(base_ch + value_ch, base_ch, kernel_size=1, padding=0, bias=True),
                nn.BatchNorm2d(base_ch),
                nn.ReLU(),
                nn.Conv2d(base_ch, out_ch, kernel_size=1, padding=0, bias=True)
            )

    def _reset_parameters(self):
        for head in self.heads:
            for m in self.base_queries_modules[head].modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
            for m in self.center_key_modules[head].modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
            for m in self.center_value_modules[head].modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
            for m in self.corner_key_modules[head].modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
            for m in self.corner_value_modules[head].modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
            for m in self.fg_key_modules[head].modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
            for m in self.fg_value_modules[head].modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
            if ('center_map' in head) or ('corner_map' in head) or ('foreground_map' in head):
                self.predict_modules[head][-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.predict_modules[head].modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

    def forward(self, base_feats, fg_feat, ctr_feat, cnr_feat):
        ret = {}
        for head in self.heads:
            base_q = self.base_queries_modules[head](base_feats)

            # center_k = self.center_key_modules[head](ctr_feat)
            # center_v = self.center_value_modules[head](ctr_feat)
            # center_a = (base_q * center_k).sum(1, keepdim=True)  # (B, 1, W, H)
            #
            # corner_k = self.corner_key_modules[head](cnr_feat)
            # corner_v = self.corner_value_modules[head](cnr_feat)
            # corner_a = (base_q * corner_k).sum(1, keepdim=True)  # (B, 1, W, H)
            #
            # fg_k = self.fg_key_modules[head](fg_feat)
            # fg_v = self.fg_value_modules[head](fg_feat)
            # fg_a = (base_q * fg_k).sum(1, keepdim=True)  # (B, 1, W, H)
            #
            # attention_weights = F.softmax(torch.cat([center_a, corner_a, fg_a], dim=1), dim=1).unsqueeze(2)
            # attention_feats = torch.stack([center_v, corner_v, fg_v], dim=1)  # (B, 3, C, H, W)
            # attention_feats = (attention_weights * attention_feats).sum(1)  # (B, C, H, W)

            center_v = self.center_value_modules[head](ctr_feat)
            corner_v = self.corner_value_modules[head](cnr_feat)
            fg_v = self.fg_value_modules[head](fg_feat)
            attention_feats = center_v + corner_v + fg_v

            final_feats = torch.cat([base_feats, attention_feats], dim=1)
            pred_feats = self.predict_modules[head](final_feats)
            ret[head] = pred_feats

        return ret


class AttnDCNFusionHead(nn.Module):
    def __init__(self, in_ch, num_cls, heads, feat_ch=64, key_ch=16, value_ch=16, init_bias=-2.19, **kwargs):
        super(AttnDCNFusionHead, self).__init__()

        # heatmap prediction head
        self.center_conv = Sequential(
            nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True)
        )
        self.center_head = nn.Conv2d(feat_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)
        self.center_head.bias.data.fill_(init_bias)

        # corner_map prediction head
        self.corner_conv = Sequential(
            nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True)
        )
        self.corner_head = nn.Conv2d(feat_ch, num_cls * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.corner_head.bias.data.fill_(init_bias)

        self.fg_conv = Sequential(
            nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(feat_ch, num_cls, kernel_size=3, stride=1, padding=1, bias=True)
        self.fg_head.bias.data.fill_(init_bias)

        # other regression target
        self.task_head = AttnSepHead(
            base_ch=in_ch, feat_ch=feat_ch, heads=heads,
            key_ch=key_ch, value_ch=value_ch, init_bias=init_bias
        )

        self.fusion_module = DCNFusionV3(feat_ch, in_ch, feat_ch)

    def forward(self, x):
        center_feat = self.center_conv(x)
        center_map = self.center_head(center_feat)

        corner_feat = self.corner_conv(x)
        corner_map = self.corner_head(corner_feat)

        fg_feat = self.fg_conv(x)
        fg_map = self.fg_head(fg_feat)
        fg_feat, ctr_feat, cnr_feat = self.fusion_module(
            orig_feat=x,
            center_map=center_map,
            center_feat=center_feat,
            corner_map=corner_map,
            corner_feat=corner_feat,
            foreground_feat=fg_feat,
            foreground_map=fg_map
        )

        ret = self.task_head(x, fg_feat, ctr_feat, cnr_feat)

        ret['center_map'] = center_map
        ret['corner_map'] = corner_map
        ret['foreground_map'] = fg_map
        return ret
