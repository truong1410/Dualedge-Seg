"""
DualEdge_Swin.py
================
Body encoder  : Swin Transformer Tiny (pretrained ImageNet)
Edge encoders : PiDiNet + DexiNed (parallel, 4-scale)
Fusion        : Multi-scale edge fusion at EACH of 4 scales (same as DualEdge_UNet)
Inject        : Attention gate at each scale → inject into Swin layers
Decoder       : DLF cross-attention + ConvUp (same as BEFUnet original)

Key difference vs DualEdge Original:
  - DualEdge Original: fuse edge ONCE at end, skip-add into Swin
  - DualEdge_Swin:     fuse edge at EACH scale, inject via attention gate into Swin

Key difference vs DualEdge_UNet:
  - DualEdge_UNet:  ResNet-50 body + U-Net decoder
  - DualEdge_Swin:  Swin Transformer body + DLF decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_

from models.Encoder import BasicLayer, PatchMerging, MultiScaleBlock
from models.Decoder import ConvUpsample, SegmentationHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DexiNed encoder
# ─────────────────────────────────────────────────────────────────────────────

class DexiNedBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1,  dilation=1)
        self.c2 = nn.Conv2d(in_ch, out_ch, 3, padding=2,  dilation=2)
        self.c3 = nn.Conv2d(in_ch, out_ch, 3, padding=4,  dilation=4)
        self.c4 = nn.Conv2d(in_ch, out_ch, 3, padding=8,  dilation=8)
        self.bn     = nn.BatchNorm2d(out_ch * 4)
        self.relu   = nn.ReLU(inplace=True)
        self.reduce = nn.Conv2d(out_ch * 4, out_ch, 1)

    def forward(self, x):
        return self.reduce(self.relu(self.bn(
            torch.cat([self.c1(x), self.c2(x), self.c3(x), self.c4(x)], 1))))


class DexiNedEncoder(nn.Module):
    def __init__(self, in_ch=3, base=30):
        super().__init__()
        self.init  = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.s1    = nn.Sequential(*[DexiNedBlock(base,   base  ) for _ in range(4)])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.s2    = nn.Sequential(DexiNedBlock(base,   base*2),
                                   *[DexiNedBlock(base*2, base*2) for _ in range(3)])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.s3    = nn.Sequential(DexiNedBlock(base*2, base*4),
                                   *[DexiNedBlock(base*4, base*4) for _ in range(3)])
        self.pool3 = nn.MaxPool2d(2, 2)
        self.s4    = nn.Sequential(*[DexiNedBlock(base*4, base*4) for _ in range(4)])

    def forward(self, x):
        x  = self.init(x)
        f1 = self.s1(x)
        f2 = self.s2(self.pool1(f1))
        f3 = self.s3(self.pool2(f2))
        f4 = self.s4(self.pool3(f3))
        return f1, f2, f3, f4


# ─────────────────────────────────────────────────────────────────────────────
# 2. PiDiNet encoder
# ─────────────────────────────────────────────────────────────────────────────

class PiDiNetEncoder(nn.Module):
    def __init__(self, config, pretrained_path=None):
        super().__init__()
        from models.pidinet import PiDiNet
        from models.config  import config_model_converted
        import collections

        pdcs    = config_model_converted(config.pdcs)
        pidinet = PiDiNet(30, pdcs, dil=12, sa=True, convert=True)

        if pretrained_path:
            ckpt     = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            sd       = ckpt['state_dict']
            new_sd   = collections.OrderedDict((k[7:], v) for k, v in sd.items())
            model_sd = pidinet.state_dict()
            filtered = {k: v for k, v in new_sd.items()
                        if k in model_sd and v.shape == model_sd[k].shape}
            model_sd.update(filtered)
            pidinet.load_state_dict(model_sd)
            print(f"✓ PiDiNet loaded {len(filtered)}/{len(new_sd)} keys")

        layers      = list(pidinet.children())
        self.stage0 = nn.Sequential(*layers[:4])
        self.stage1 = nn.Sequential(*layers[4:8])
        self.stage2 = nn.Sequential(*layers[8:12])
        self.stage3 = nn.Sequential(*layers[12:16])

    def forward(self, x):
        f1 = self.stage0(x)
        f2 = self.stage1(f1)
        f3 = self.stage2(f2)
        f4 = self.stage3(f3)
        return f1, f2, f3, f4


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-scale edge fusion (same as DualEdge_UNet)
# ─────────────────────────────────────────────────────────────────────────────

class ScaleFusion(nn.Module):
    """Fuse PiDiNet + DexiNed at one scale."""
    def __init__(self, channels, fusion_type='weighted_sum', window_size=7):
        super().__init__()
        self.fusion_type = fusion_type
        self.window_size = window_size

        if fusion_type == 'concat':
            self.mix = nn.Sequential(
                nn.Conv2d(channels*2, channels, 1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        elif fusion_type == 'weighted_sum':
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.beta  = nn.Parameter(torch.tensor(0.5))
            self.mix   = nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        else:  # cross_attention
            r = max(channels // 2, 16)
            h = max(1, min(4, r // 8))
            self.reduce  = nn.Conv2d(channels, r, 1, bias=False)
            self.expand  = nn.Conv2d(r, channels, 1, bias=False)
            self.attn_pd = nn.MultiheadAttention(r, h, batch_first=True, dropout=0.)
            self.attn_dp = nn.MultiheadAttention(r, h, batch_first=True, dropout=0.)
            self.mix     = nn.Sequential(
                nn.Conv2d(channels*2, channels, 1, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def _win_part(self, x, win):
        B, r, H, W = x.shape
        x  = x.permute(0,2,3,1)
        ph = (win - H%win)%win; pw = (win - W%win)%win
        if ph or pw: x = F.pad(x, (0,0,0,pw,0,ph))
        _, Hp, Wp, _ = x.shape
        x = x.reshape(B, Hp//win, win, Wp//win, win, r)
        x = x.permute(0,1,3,2,4,5).reshape(-1, win*win, r)
        return x, Hp, Wp

    def _win_rev(self, x, win, B, Hp, Wp, H, W, r):
        x = x.reshape(B, Hp//win, Wp//win, win, win, r)
        x = x.permute(0,5,1,3,2,4).reshape(B, r, Hp, Wp)
        return x[:,:,:H,:W]

    def forward(self, fp, fd):
        if self.fusion_type == 'concat':
            return self.mix(torch.cat([fp, fd], dim=1))
        elif self.fusion_type == 'weighted_sum':
            a = torch.sigmoid(self.alpha); b = torch.sigmoid(self.beta)
            t = a + b + 1e-8
            return self.mix(a/t * fp + b/t * fd)
        else:
            B, C, H, W = fp.shape
            win  = self.window_size
            fp_r = self.reduce(fp); fd_r = self.reduce(fd)
            fp_w, Hp, Wp = self._win_part(fp_r, win)
            fd_w, _,  _  = self._win_part(fd_r, win)
            fp_a, _ = self.attn_pd(fp_w, fd_w, fd_w); fp_a = fp_a + fp_w
            fd_a, _ = self.attn_dp(fd_w, fp_w, fp_w); fd_a = fd_a + fd_w
            r = fp_r.shape[1]
            fp_o = self.expand(self._win_rev(fp_a, win, B, Hp, Wp, H, W, r))
            fd_o = self.expand(self._win_rev(fd_a, win, B, Hp, Wp, H, W, r))
            return self.mix(torch.cat([fp_o, fd_o], dim=1))


class MultiScaleEdgeFusion(nn.Module):
    def __init__(self, channels, fusion_type='weighted_sum'):
        super().__init__()
        self.fusions = nn.ModuleList([
            ScaleFusion(c, fusion_type) for c in channels])

    def forward(self, pidi_feats, dexi_feats):
        return [self.fusions[i](pidi_feats[i], dexi_feats[i]) for i in range(4)]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Swin Transformer body
# ─────────────────────────────────────────────────────────────────────────────

class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=96,
                 depths=[2,2,6,2], num_heads=[3,6,12,24],
                 window_size=7, mlp_ratio=4., drop_path_rate=0.1):
        super().__init__()
        patches_resolution = [img_size//patch_size, img_size//patch_size]
        self.num_layers = len(depths)
        self.embed_dim  = embed_dim
        self.pos_drop   = nn.Dropout(p=0.)
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.layers = nn.ModuleList()
        dpr_ptr = 0
        for i in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i),
                input_resolution=(patches_resolution[0]//(2**i),
                                   patches_resolution[1]//(2**i)),
                depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio,
                drop_path=dpr[dpr_ptr:dpr_ptr+depths[i]], downsample=None)
            dpr_ptr += depths[i]
            self.layers.append(layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.zeros_(m.bias)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Edge → Swin injection via attention gate (per scale)
# ─────────────────────────────────────────────────────────────────────────────

class EdgeSwinInject(nn.Module):
    """
    Inject edge features into Swin token sequence via attention gate.
    edge: [B, edge_ch, H, W]  (spatial)
    swin: [B, L, swin_ch]     (sequence)
    output: [B, L, swin_ch]
    """
    def __init__(self, swin_ch, edge_ch):
        super().__init__()
        self.edge_proj = nn.Conv2d(edge_ch, swin_ch, 1, bias=False)
        self.attn      = nn.Sequential(
            nn.Conv2d(swin_ch*2, swin_ch, 1, bias=False),
            nn.BatchNorm2d(swin_ch), nn.ReLU(inplace=True),
            nn.Conv2d(swin_ch, 1, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(swin_ch)

    def forward(self, swin_seq, edge, H, W):
        """
        swin_seq: [B, H*W, C]
        edge:     [B, edge_ch, H', W']
        """
        B, L, C = swin_seq.shape
        # Reshape swin to spatial
        swin_sp = swin_seq.transpose(1,2).reshape(B, C, H, W)
        # Project edge to swin channels
        if edge.shape[2:] != (H, W):
            edge = F.interpolate(edge, (H, W), mode='bilinear', align_corners=False)
        ep   = self.edge_proj(edge)
        # Attention gate
        attn = self.attn(torch.cat([swin_sp, ep], dim=1))
        # Inject
        fused = swin_sp + ep * attn
        # Back to sequence
        fused_seq = fused.reshape(B, C, L).transpose(1, 2)
        return self.norm(fused_seq)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pyramid Features: multi-scale edge + Swin
# ─────────────────────────────────────────────────────────────────────────────

EDGE_CH = [30, 60, 120, 120]

class PyramidFeatures_DualEdge_Swin(nn.Module):
    def __init__(self, config, img_size=224, in_channels=3, fusion_type='weighted_sum'):
        super().__init__()

        # Swin
        model_path = config.swin_pretrained_path
        self.swin  = SwinTransformer(img_size)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)['model']

        unexpected = [
            "patch_embed.proj.weight","patch_embed.proj.bias",
            "patch_embed.norm.weight","patch_embed.norm.bias",
            "head.weight","head.bias",
            "layers.0.downsample.norm.weight","layers.0.downsample.norm.bias",
            "layers.0.downsample.reduction.weight",
            "layers.1.downsample.norm.weight","layers.1.downsample.norm.bias",
            "layers.1.downsample.reduction.weight",
            "layers.2.downsample.norm.weight","layers.2.downsample.norm.bias",
            "layers.2.downsample.reduction.weight",
            "layers.3.downsample.norm.weight","layers.3.downsample.norm.bias",
            "layers.3.downsample.reduction.weight",
            "norm.weight","norm.bias",
        ]

        # Edge encoders
        self.pidi = PiDiNetEncoder(config,
            pretrained_path=config.PDC_pretrained_path if config.pidinet_pretrained else None)
        self.dexi = DexiNedEncoder(in_ch=in_channels, base=30)

        # Multi-scale edge fusion
        self.edge_fusion = MultiScaleEdgeFusion(EDGE_CH, fusion_type)

        # Edge→Swin injection at each of 4 Swin levels
        # Swin dims: [96, 192, 384, 768]
        swin_dims = [int(96 * 2**i) for i in range(4)]
        self.injectors = nn.ModuleList([
            EdgeSwinInject(swin_dims[i], EDGE_CH[i]) for i in range(4)
        ])

        # PatchMerging layers (load pretrained weights)
        self.p1_pm = PatchMerging(
            (img_size//4, img_size//4), 96)
        self.p2_pm = PatchMerging(
            (img_size//8, img_size//8), 192)
        self.p3_pm = PatchMerging(
            (img_size//16, img_size//16), 384)

        self.p1_pm.state_dict()['reduction.weight'][:] = checkpoint["layers.0.downsample.reduction.weight"]
        self.p1_pm.state_dict()['norm.weight'][:]      = checkpoint["layers.0.downsample.norm.weight"]
        self.p1_pm.state_dict()['norm.bias'][:]        = checkpoint["layers.0.downsample.norm.bias"]
        self.p2_pm.state_dict()['reduction.weight'][:] = checkpoint["layers.1.downsample.reduction.weight"]
        self.p2_pm.state_dict()['norm.weight'][:]      = checkpoint["layers.1.downsample.norm.weight"]
        self.p2_pm.state_dict()['norm.bias'][:]        = checkpoint["layers.1.downsample.norm.bias"]
        self.p3_pm.state_dict()['reduction.weight'][:] = checkpoint["layers.2.downsample.reduction.weight"]
        self.p3_pm.state_dict()['norm.weight'][:]      = checkpoint["layers.2.downsample.norm.weight"]
        self.p3_pm.state_dict()['norm.bias'][:]        = checkpoint["layers.2.downsample.norm.bias"]

        # Patch embed (project to Swin embed space)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=4, stride=4),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(96))

        self.norm_1    = nn.LayerNorm(96)
        self.norm_2    = nn.LayerNorm(768)
        self.avgpool_1 = nn.AdaptiveAvgPool1d(1)
        self.avgpool_2 = nn.AdaptiveAvgPool1d(1)

        for key in list(checkpoint.keys()):
            if key in unexpected: del checkpoint[key]
        self.swin.load_state_dict(checkpoint)

    def forward(self, x):
        B = x.shape[0]

        # Edge features at 4 scales
        pidi_feats = self.pidi(x)
        dexi_feats = self.dexi(x)
        edge_feats = self.edge_fusion(pidi_feats, dexi_feats)
        # edge_feats[i]: [B, EDGE_CH[i], H_i, W_i]

        # Patch embed → Swin Level 0
        tok = self.patch_embed(x)           # [B, H/4*W/4, 96]
        H0, W0 = x.shape[2]//4, x.shape[3]//4

        # Level 0: Swin layer 0 + edge inject
        sw0  = self.swin.layers[0](tok)
        sw0  = self.injectors[0](sw0, edge_feats[0], H0, W0)
        norm1 = self.norm_1(sw0)
        cls1  = Rearrange('b c 1 -> b 1 c')(self.avgpool_1(norm1.transpose(1,2)))
        sw0_pm = self.p1_pm(sw0)            # [B, H/8*W/8, 192]

        # Level 1
        H1, W1 = H0//2, W0//2
        sw1  = self.swin.layers[1](sw0_pm)
        sw1  = self.injectors[1](sw1, edge_feats[1], H1, W1)
        sw1_pm = self.p2_pm(sw1)            # [B, H/16*W/16, 384]

        # Level 2
        H2, W2 = H1//2, W1//2
        sw2  = self.swin.layers[2](sw1_pm)
        sw2  = self.injectors[2](sw2, edge_feats[2], H2, W2)
        sw2_pm = self.p3_pm(sw2)            # [B, H/32*W/32, 768]

        # Level 3
        H3, W3 = H2//2, W2//2
        sw3  = self.swin.layers[3](sw2_pm)
        sw3  = self.injectors[3](sw3, edge_feats[3], H3, W3)
        norm2 = self.norm_2(sw3)
        cls4  = Rearrange('b c 1 -> b 1 c')(self.avgpool_2(norm2.transpose(1,2)))

        return [
            torch.cat((cls1, sw0), dim=1),   # [B, 1+L0, 96]
            torch.cat((cls4, sw3), dim=1),   # [B, 1+L3, 768]
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 7. All2Cross + DLF (reuse from BEFUnet)
# ─────────────────────────────────────────────────────────────────────────────

class All2Cross_DualEdge_Swin(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3,
                 embed_dim=(96,768), norm_layer=nn.LayerNorm,
                 fusion_type='weighted_sum'):
        super().__init__()
        self.cross_pos_embed = config.cross_pos_embed
        self.pyramid = PyramidFeatures_DualEdge_Swin(
            config, img_size, in_chans, fusion_type)

        self.num_branches = 2
        n_p1 = (config.image_size // config.patch_size    ) ** 2
        n_p2 = (config.image_size // config.patch_size // 8) ** 2
        num_patches = (n_p1, n_p2)

        self.pos_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1+num_patches[i], embed_dim[i]))
            for i in range(self.num_branches)])

        total_depth = sum(sum(x[-2:]) for x in config.depth)
        dpr = torch.linspace(0, config.drop_path_rate, total_depth).tolist()
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for block_config in config.depth:
            curr_depth = max(block_config[:-1]) + block_config[-1]
            blk = MultiScaleBlock(
                embed_dim, num_patches, block_config,
                num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                drop=config.drop_rate, attn_drop=config.attn_drop_rate,
                drop_path=dpr[dpr_ptr:dpr_ptr+curr_depth], norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias); nn.init.ones_(m.weight)

    def forward(self, x):
        xs = self.pyramid(x)
        if self.cross_pos_embed:
            for i in range(self.num_branches):
                xs[i] = xs[i] + self.pos_embed[i]
        for blk in self.blocks:
            xs = blk(xs)
        return [self.norm[i](xs[i]) for i in range(self.num_branches)]


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main model
# ─────────────────────────────────────────────────────────────────────────────

class DualEdge_Swin(nn.Module):
    """
    DualEdge_Swin:
      PiDiNet + DexiNed → multi-scale edge fusion (4 scales)
      Swin Transformer   → body with attention-gate edge injection per scale
      DLF cross-attention → shallow + deep feature fusion
      ConvUp decoder     → same as BEFUnet original
    """

    def __init__(self, config, n_classes=9, fusion_type='weighted_sum', img_size=224):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = [4, 32]
        self.n_classes  = n_classes
        self.fusion_type = fusion_type

        self.All2Cross = All2Cross_DualEdge_Swin(
            config, img_size, in_chans=3,
            fusion_type=fusion_type)

        self.ConvUp_s = ConvUpsample(in_chans=768, out_chans=[128,128,128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=96,  upsample=False)

        self.segmentation_head = SegmentationHead(
            in_channels=16, out_channels=n_classes, kernel_size=3)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False))

    def forward(self, x):
        xs         = self.All2Cross(x)
        embeddings = [xi[:, 1:] for xi in xs]
        reshaped   = []
        for i, embed in enumerate(embeddings):
            embed = Rearrange(
                'b (h w) d -> b d h w',
                h=self.img_size // self.patch_size[i],
                w=self.img_size // self.patch_size[i])(embed)
            embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)
            reshaped.append(embed)
        C   = reshaped[0] + reshaped[1]
        C   = self.conv_pred(C)
        return self.segmentation_head(C)
