"""
K-Fold Cross Validation training for DualEdge BEFUnet
- 30 cases total (18 train + 12 test) split into 5 folds
- Each fold: 24 cases train, 6 cases validation
- Test set (12 cases) stays fixed for final evaluation
- Trains 3 fusion variants: WeightedSum, CrossAttention, Concat
"""

import argparse
import os
import random
import sys
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import h5py
from scipy.ndimage import zoom

np.bool = bool; np.int = int; np.float = float
np.complex = complex; np.object = object; np.str = str

import configs.BEFUnet_DualEdge_configs as cfg_module
from losses.boundary_loss import CombinedBoundaryLoss


# ── All 30 cases ─────────────────────────────────────────────────────────────
ALL_CASES = [
    # 18 original train cases
    'case0005','case0006','case0007','case0009','case0010',
    'case0021','case0023','case0024','case0026','case0027',
    'case0028','case0030','case0031','case0033','case0034',
    'case0037','case0039','case0040',
    # 12 original test cases
    'case0001','case0002','case0003','case0004','case0008',
    'case0022','case0025','case0029','case0032','case0035',
    'case0036','case0038',
]

# Fixed test set (never used in train/val)
FIXED_TEST = [
    'case0001','case0002','case0003','case0004','case0008',
    'case0022','case0025','case0029','case0032','case0035',
    'case0036','case0038',
]

# 18 cases available for k-fold
KFOLD_CASES = [c for c in ALL_CASES if c not in FIXED_TEST]


def make_5_folds(cases, seed=1234):
    """Split 18 cases into 5 folds (~3-4 cases each)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(cases))
    shuffled = [cases[i] for i in idx]
    folds = []
    for k in range(5):
        folds.append(shuffled[k::5])
    return folds   # list of 5 lists


# ── Dataset ───────────────────────────────────────────────────────────────────

class SynapseSliceDataset(Dataset):
    """Load slices from .npz files for a given list of cases."""

    def __init__(self, cases, npz_dir, img_size=224, augment=True):
        self.img_size = img_size
        self.augment  = augment
        self.samples  = []
        for case in cases:
            slices = sorted([
                f for f in os.listdir(npz_dir)
                if f.startswith(case + '_slice') and f.endswith('.npz')
            ])
            for s in slices:
                self.samples.append(os.path.join(npz_dir, s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data  = np.load(self.samples[idx])
        image = data['image'].astype(np.float32)  # [H, W]
        label = data['label'].astype(np.int64)    # [H, W]

        H, W = image.shape
        if H != self.img_size or W != self.img_size:
            image = zoom(image, (self.img_size/H, self.img_size/W), order=3)
            label = zoom(label, (self.img_size/H, self.img_size/W), order=0)

        if self.augment:
            if random.random() > 0.5:
                image = np.flip(image, 0).copy(); label = np.flip(label, 0).copy()
            if random.random() > 0.5:
                image = np.flip(image, 1).copy(); label = np.flip(label, 1).copy()
            if random.random() > 0.5:
                k = random.randint(1, 3)
                image = np.rot90(image, k).copy(); label = np.rot90(label, k).copy()

        # Replicate grayscale to 3 channels
        image = np.stack([image, image, image], axis=0)   # [3, H, W]
        return {
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(label.copy()).long(),
        }


class SynapseVolumeDataset(Dataset):
    """Load full volumes from .h5 files for validation."""

    def __init__(self, cases, h5_dir, img_size=224):
        self.img_size = img_size
        self.h5_dir   = h5_dir
        self.cases    = cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        # Try both .npy.h5 and plain .h5
        for ext in ['.npy.h5', '.h5']:
            path = os.path.join(self.h5_dir, case + ext)
            if os.path.exists(path):
                break
        # If not in h5_dir, build from npz dir (train cases)
        if not os.path.exists(path):
            return {'image': None, 'label': None, 'case_name': case}
        with h5py.File(path, 'r') as f:
            image = f['image'][:]
            label = f['label'][:]
        return {'image': torch.from_numpy(image).unsqueeze(0),
                'label': torch.from_numpy(label).unsqueeze(0),
                'case_name': case}


def collate_fn(batch):
    return {
        'image':     torch.stack([s['image'] for s in batch], 0),
        'label':     torch.stack([s['label'] for s in batch], 0),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def dice_score(pred, gt, n_classes):
    pred = torch.argmax(torch.softmax(pred.detach(), dim=1), dim=1)
    dices = []
    for c in range(1, n_classes):
        p = (pred == c).float(); g = (gt == c).float()
        inter = (p * g).sum(); union = p.sum() + g.sum()
        if union > 0:
            dices.append((2*inter / union).item())
    return np.mean(dices) if dices else 0.0


# ── Single fold training ──────────────────────────────────────────────────────

def train_one_fold(args, fold_idx, train_cases, val_cases, fusion_type):
    """Train one fold for one fusion type."""

    out_dir = os.path.join(
        args.output_dir, f'fusion_{fusion_type}', f'fold_{fold_idx}')
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, 'train.log')
    logger   = logging.getLogger(f'{fusion_type}_fold{fold_idx}')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', '%H:%M:%S'))
        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info(f"Fold {fold_idx} | Fusion: {fusion_type}")
    logger.info(f"Train cases ({len(train_cases)}): {train_cases}")
    logger.info(f"Val   cases ({len(val_cases)}):   {val_cases}")

    # Build model with selected fusion
    config = cfg_module.get_BEFUnet_DualEdge_configs()

    # Patch backbone to use correct fusion
    import models.dualedge_backbone as bb
    if fusion_type == 'weighted_sum':
        fusion_cls = bb.DualEdgeFusion   # original weighted sum
    elif fusion_type == 'cross_attention':
        fusion_cls = bb.CrossAttentionFusion
    else:
        fusion_cls = bb.ConcatFusion

    # Temporarily monkey-patch DualEdgeBackbone to use chosen fusion
    orig_init = bb.DualEdgeBackbone.__init__

    def patched_init(self, config, in_channels=3):
        orig_init(self, config, in_channels)
        self.fuse = fusion_cls(channels=config.cnn_pyramid_fm)

    bb.DualEdgeBackbone.__init__ = patched_init

    from models.BEFUnet_DualEdge import BEFUnet_DualEdge
    model = BEFUnet_DualEdge(config, img_size=args.img_size,
                             in_chans=3, n_classes=args.num_classes).cuda()
    bb.DualEdgeBackbone.__init__ = orig_init  # restore

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Parameters: {n_params:.2f}M")

    # Datasets
    train_ds = SynapseSliceDataset(
        train_cases, args.npz_dir, args.img_size, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # Optimizer & scheduler
    criterion = CombinedBoundaryLoss(n_classes=args.num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6)

    best_dice   = -1.0
    start_epoch = 1

    # Resume from latest checkpoint if exists
    resume_path = os.path.join(out_dir, 'latest_checkpoint.pth')
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location='cuda', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice   = ckpt.get('best_dice', -1.0)
        logger.info(f"✓ Resumed from epoch {ckpt['epoch']} (best_dice={best_dice:.4f})")

    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()
        epoch_loss = 0.; epoch_dice = 0.; n_batches = 0

        pbar = tqdm(train_loader,
                    desc=f"[{fusion_type}] Fold{fold_idx} Ep{epoch}/{args.max_epochs}")
        for batch in pbar:
            images = batch['image'].cuda()
            labels = batch['label'].cuda()
            outputs = model(images)
            loss, _ = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            d = dice_score(outputs, labels, args.num_classes)
            epoch_loss += loss.item(); epoch_dice += d; n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{d:.4f}'})

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        avg_dice = epoch_dice / n_batches
        logger.info(f"Epoch {epoch}/{args.max_epochs} | Loss {avg_loss:.4f} | Dice {avg_dice:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'loss': avg_loss,
                'fusion_type': fusion_type,
                'fold': fold_idx,
                'train_cases': train_cases,
                'val_cases': val_cases,
            }, os.path.join(out_dir, 'best_model.pth'))
            logger.info(f"  ✓ Best model saved (Dice={best_dice:.4f})")

        # Save latest checkpoint every epoch for resume
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'loss': avg_loss,
            'fusion_type': fusion_type,
            'fold': fold_idx,
        }, os.path.join(out_dir, 'latest_checkpoint.pth'))

        if epoch % args.eval_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_dice': avg_dice,
                'loss': avg_loss,
            }, os.path.join(out_dir, f'epoch_{epoch}.pth'))

    logger.info(f"Fold {fold_idx} done. Best Dice: {best_dice:.4f}")
    return best_dice


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir',      default='./data/Synapse/train_npz')
    parser.add_argument('--h5_dir',       default='./data/Synapse/test_vol_h5')
    parser.add_argument('--list_dir',     default='./lists/lists_Synapse')
    parser.add_argument('--num_classes',  type=int,   default=9)
    parser.add_argument('--max_epochs',   type=int,   default=400)
    parser.add_argument('--batch_size',   type=int,   default=10)
    parser.add_argument('--base_lr',      type=float, default=1e-4)
    parser.add_argument('--img_size',     type=int,   default=224)
    parser.add_argument('--seed',         type=int,   default=1234)
    parser.add_argument('--n_folds',      type=int,   default=5)
    parser.add_argument('--output_dir',   default='./results_kfold')
    parser.add_argument('--eval_interval',type=int,   default=20)
    parser.add_argument('--fusion_types', nargs='+',
                        default=['weighted_sum', 'cross_attention', 'concat'],
                        help='Fusion types to train')
    parser.add_argument('--folds',        nargs='+',  type=int,
                        default=None,
                        help='Which folds to run (default: all)')
    args = parser.parse_args()

    cudnn.benchmark = False; cudnn.deterministic = True
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Create folds
    folds = make_5_folds(KFOLD_CASES, seed=args.seed)

    print("\n" + "="*60)
    print(f"K-Fold Cross Validation  (K={args.n_folds})")
    print(f"Total cases for CV: {len(KFOLD_CASES)}")
    print(f"Fixed test cases  : {len(FIXED_TEST)}")
    print(f"Fusion types      : {args.fusion_types}")
    print("="*60)
    for i, fold in enumerate(folds):
        print(f"  Fold {i}: {fold}")
    print("="*60 + "\n")

    # Save fold assignments
    import json
    fold_info = {f'fold_{i}': folds[i] for i in range(len(folds))}
    with open(os.path.join(args.output_dir, 'fold_assignments.json'), 'w') as f:
        json.dump(fold_info, f, indent=2)

    # Run training
    run_folds = args.folds if args.folds else list(range(args.n_folds))
    results = {}

    for fusion_type in args.fusion_types:
        results[fusion_type] = {}
        for fold_idx in run_folds:
            val_cases   = folds[fold_idx]
            train_cases = [c for i, f in enumerate(folds)
                          for c in f if i != fold_idx]

            print(f"\n{'='*60}")
            print(f"Fusion: {fusion_type}  |  Fold {fold_idx}")
            print(f"Train: {train_cases}")
            print(f"Val  : {val_cases}")
            print('='*60)

            best_dice = train_one_fold(
                args, fold_idx, train_cases, val_cases, fusion_type)
            results[fusion_type][f'fold_{fold_idx}'] = best_dice

    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for fusion_type, fold_results in results.items():
        dices = list(fold_results.values())
        print(f"\n{fusion_type}:")
        for fold, d in fold_results.items():
            print(f"  {fold}: {d:.4f}")
        print(f"  Mean: {np.mean(dices):.4f} ± {np.std(dices):.4f}")

    # Save results
    with open(os.path.join(args.output_dir, 'kfold_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {args.output_dir}/kfold_results.json")
