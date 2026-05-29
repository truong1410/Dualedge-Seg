"""
Topology-Guided Learning for Medical Image Segmentation
========================================================

Key components:
  1. Backbone: ResNet-50 + U-Net decoder (reuse from DualEdge_UNet)
  2. Topology Module: Persistent Homology via differentiable filtration
  3. Topology Loss: match persistence diagrams of pred vs GT
  4. Combined loss: CE + Dice + λ * Topology Loss

Persistent Homology basics:
  - Filtration: threshold prob map at t=0,0.01,...,1.0
  - Track connected components (H0) and holes (H1) birth/death
  - Persistence diagram: set of (birth, death) pairs
  - Topology loss: Wasserstein distance between pred and GT diagrams

Note: Full persistent homology requires gudhi/ripser library.
      We implement a differentiable approximation using:
      - Euler Characteristic as H0 proxy
      - Morphological operations for connectivity analysis
      - Differentiable persistence via soft-thresholding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Backbone: ResNet-50 U-Net (reuse from DualEdge_UNet)
# ─────────────────────────────────────────────────────────────────────────────

class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # [B, 256,  H/4,  W/4]
        self.layer2 = resnet.layer2   # [B, 512,  H/8,  W/8]
        self.layer3 = resnet.layer3   # [B, 1024, H/16, W/16]
        self.layer4 = resnet.layer4   # [B, 2048, H/32, W/32]

    def forward(self, x):
        x  = self.stem(x)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return s1, s2, s3, s4


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch+skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.dec4 = DecoderBlock(2048, 1024, 512)
        self.dec3 = DecoderBlock(512,   512, 256)
        self.dec2 = DecoderBlock(256,   256, 128)
        self.dec1 = DecoderBlock(128,     0,  64)
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 1))

    def forward(self, s1, s2, s3, s4):
        x = self.dec4(s4, s3)
        x = self.dec3(x,  s2)
        x = self.dec2(x,  s1)
        x = self.dec1(x,  None)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return self.seg_head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Topology Module: Differentiable Persistent Homology approximation
# ─────────────────────────────────────────────────────────────────────────────

class SoftFiltration(nn.Module):
    """
    Differentiable filtration of probability map.

    For each threshold t in [0, 1]:
      - Compute soft binary map: M_t = sigmoid((prob - t) / temperature)
      - Count approximate connected components via Euler characteristic
      - Track birth/death of topological features

    Euler Characteristic (χ) = V - E + F (vertices - edges + faces)
    For 2D binary image: χ ≈ n_components - n_holes (approximate H0-H1)

    We approximate χ differentiably using local pixel statistics.
    """
    def __init__(self, n_thresholds=20, temperature=0.05):
        super().__init__()
        self.n_thresholds = n_thresholds
        self.temperature  = temperature
        thresholds = torch.linspace(0.05, 0.95, n_thresholds)
        self.register_buffer('thresholds', thresholds)

    def euler_characteristic_soft(self, soft_mask):
        """
        Approximate Euler characteristic of soft binary mask.
        soft_mask: [B, H, W] in [0, 1]

        Uses pixel, edge, and face counts:
        V = sum of pixels
        E = sum of adjacent pixel pairs (H and V edges)
        F = sum of 2x2 pixel quads
        χ = V - E + F
        """
        B, H, W = soft_mask.shape
        x = soft_mask

        # Vertices
        V = x.sum(dim=(1,2))

        # Horizontal edges: min of adjacent pairs
        E_h = torch.minimum(x[:, :, :-1], x[:, :, 1:]).sum(dim=(1,2))
        # Vertical edges
        E_v = torch.minimum(x[:, :-1, :], x[:, 1:, :]).sum(dim=(1,2))
        E   = E_h + E_v

        # Faces (2×2 quads): min of all 4 corners
        F = torch.minimum(
            torch.minimum(x[:, :-1, :-1], x[:, :-1, 1:]),
            torch.minimum(x[:, 1:,  :-1], x[:, 1:,  1:])).sum(dim=(1,2))

        return V - E + F   # [B]

    def forward(self, prob_map):
        """
        prob_map: [B, H, W]  probability of foreground (single class)
        Returns: persistence_curve [B, n_thresholds]
                 (Euler characteristic at each threshold)
        """
        B = prob_map.shape[0]
        euler_curve = []

        for t in self.thresholds:
            # Soft thresholding
            soft_mask = torch.sigmoid((prob_map - t) / self.temperature)
            ec = self.euler_characteristic_soft(soft_mask)  # [B]
            euler_curve.append(ec)

        return torch.stack(euler_curve, dim=1)   # [B, n_thresholds]


class PersistenceDiagramApprox(nn.Module):
    """
    Approximate persistence diagram from Euler characteristic curve.

    The Euler characteristic curve tracks topology changes across filtration.
    Peaks and valleys correspond to topological features (components, holes).

    We extract:
      - Birth times: where EC increases (new component appears)
      - Death times: where EC decreases (components merge)
      - Persistence: death - birth

    This is a differentiable approximation — not exact persistent homology.
    For exact computation, use gudhi or ripser libraries.
    """
    def __init__(self, n_thresholds=20):
        super().__init__()
        self.n_thresholds = n_thresholds

    def forward(self, ec_curve):
        """
        ec_curve: [B, n_thresholds]
        Returns: persistence features [B, n_thresholds-1]
        """
        # Derivative of EC curve = topology change rate
        # Positive: new components born
        # Negative: components die
        delta = ec_curve[:, 1:] - ec_curve[:, :-1]   # [B, n_thresholds-1]
        return delta


# ─────────────────────────────────────────────────────────────────────────────
# 3. Topology Loss
# ─────────────────────────────────────────────────────────────────────────────

class TopologyLoss(nn.Module):
    """
    Topology loss comparing persistence diagrams of prediction and GT.

    L_topo = sum over classes of:
      || delta_pred - delta_gt ||^2  (L2 distance between EC curves)

    This penalizes:
      - Wrong number of connected components
      - Missing/extra holes in the segmentation
      - Topologically incorrect connectivity

    For exact Wasserstein distance between persistence diagrams,
    use gudhi.wasserstein or ripser. Here we use L2 as approximation.
    """
    def __init__(self, n_classes=9, n_thresholds=20, temperature=0.05):
        super().__init__()
        self.n_classes = n_classes
        self.filtration = SoftFiltration(n_thresholds, temperature)
        self.persistence = PersistenceDiagramApprox(n_thresholds)

    def forward(self, logits, label):
        """
        logits: [B, n_classes, H, W]
        label:  [B, H, W] long
        """
        prob = F.softmax(logits, dim=1)   # [B, C, H, W]
        toh  = F.one_hot(label, self.n_classes).permute(0,3,1,2).float()

        topo_loss = torch.tensor(0., device=logits.device, requires_grad=True)

        for c in range(1, self.n_classes):   # skip background
            # Only compute if class present in batch
            if (label == c).sum() == 0:
                continue

            pred_prob = prob[:, c]     # [B, H, W]
            gt_prob   = toh[:, c]     # [B, H, W]

            # Filtration → EC curve
            ec_pred = self.filtration(pred_prob)    # [B, n_thresholds]
            ec_gt   = self.filtration(gt_prob)

            # Persistence features
            pd_pred = self.persistence(ec_pred)     # [B, n_thresholds-1]
            pd_gt   = self.persistence(ec_gt)

            # L2 distance between persistence features
            topo_loss = topo_loss + F.mse_loss(pd_pred, pd_gt)

        return topo_loss / (self.n_classes - 1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Combined Loss
# ─────────────────────────────────────────────────────────────────────────────

class TopologyGuidedLoss(nn.Module):
    """
    L_total = L_CE + L_Dice + λ_topo * L_topo
    """
    def __init__(self, n_classes=9, lambda_topo=0.1,
                 n_thresholds=20, temperature=0.05, smooth=1e-5):
        super().__init__()
        self.n_classes   = n_classes
        self.lambda_topo = lambda_topo
        self.smooth      = smooth
        self.ce          = nn.CrossEntropyLoss()
        self.topo_loss   = TopologyLoss(n_classes, n_thresholds, temperature)

    def dice_loss(self, prob, label):
        toh   = F.one_hot(label, self.n_classes).permute(0,3,1,2).float()
        B,C,H,W = prob.shape
        p     = prob.view(B,C,-1); t = toh.view(B,C,-1)
        inter = (p*t).sum(2); union = p.sum(2)+t.sum(2)
        dice  = (2*inter+self.smooth)/(union+self.smooth)
        return (1-dice[:,1:].mean()).mean()

    def forward(self, logits, label):
        prob   = F.softmax(logits, dim=1)
        l_ce   = self.ce(logits, label)
        l_dice = self.dice_loss(prob, label)
        l_topo = self.topo_loss(logits, label)
        total  = l_ce + l_dice + self.lambda_topo * l_topo
        return total, {
            'total': total.item(),
            'ce':    l_ce.item(),
            'dice':  l_dice.item(),
            'topo':  l_topo.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main Model: TopologyGuidedNet
# ─────────────────────────────────────────────────────────────────────────────

class TopologyGuidedNet(nn.Module):
    """
    Topology-Guided Segmentation Network.

    Architecture:
      Input → ResNet-50 backbone → U-Net decoder → seg_logits
      seg_logits → Topology Module (filtration + persistence diagram)
      Loss: CE + Dice + Topology Loss
    """
    def __init__(self, n_classes=9, pretrained=True):
        super().__init__()
        self.n_classes = n_classes
        self.encoder   = ResNet50Encoder(pretrained)
        self.decoder   = UNetDecoder(n_classes)

    def forward(self, x):
        s1, s2, s3, s4 = self.encoder(x)
        logits          = self.decoder(s1, s2, s3, s4)
        return logits
