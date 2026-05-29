"""Training script for BoundaryAwareNet (concat and weighted_sum fusion)."""

import argparse, os, random, sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

np.bool=bool; np.int=int; np.float=float
np.complex=complex; np.object=object; np.str=str

from models.boundary_aware_net import BoundaryAwareNet
from losses.boundary_aware_loss import BoundaryAwareLoss
import configs.boundary_aware_configs as cfg_module


def dice_score_batch(logits, labels, n_classes):
    preds = torch.argmax(torch.softmax(logits.detach(), dim=1), dim=1)
    dices = []
    for c in range(1, n_classes):
        p = (preds==c).float(); g = (labels==c).float()
        inter = (p*g).sum(); union = p.sum()+g.sum()
        if union > 0: dices.append((2*inter/union).item())
    return np.mean(dices) if dices else 0.0


def collate_fn(batch):
    images = []
    for s in batch:
        img = s['image']
        if img.shape[0] == 1: img = img.repeat(3,1,1)
        images.append(img)
    return {
        'image': torch.stack(images, 0),
        'label': torch.stack([s['label'] for s in batch], 0),
        'case_name': [s.get('case_name','') for s in batch],
    }


def train_one(args, fusion_type):
    out_dir = os.path.join(args.output_dir, f'BoundaryAwareNet_{fusion_type}')
    os.makedirs(out_dir, exist_ok=True)

    config = cfg_module.get_boundary_aware_configs()
    model  = BoundaryAwareNet(
        config, n_classes=args.num_classes,
        fusion_type=fusion_type, img_size=args.img_size).cuda()

    n_params = sum(p.numel() for p in model.parameters())/1e6
    print(f"\n✓ BoundaryAwareNet [{fusion_type}]  {n_params:.2f}M params")

    criterion = BoundaryAwareLoss(
        n_classes      = args.num_classes,
        lambda1        = config.lambda1,
        lambda2        = config.lambda2,
        edge_pos_weight= config.edge_pos_weight,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6)

    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    db_train = Synapse_dataset(
        base_dir=args.root_path, list_dir=args.list_dir, split='train',
        transform=transforms.Compose([RandomGenerator([args.img_size]*2)]))
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, collate_fn=collate_fn)

    print(f"✓ Training samples: {len(db_train)}")

    best_dice   = -1.0
    start_epoch = 1

    # Resume
    resume_path = os.path.join(out_dir, 'latest_checkpoint.pth')
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location='cuda', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice   = ckpt.get('best_dice', -1.0)
        print(f"✓ Resumed from epoch {ckpt['epoch']} (best_dice={best_dice:.4f})")

    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()
        ep_loss=0; ep_dice=0; ep_edge=0; ep_bnd=0; n=0

        pbar = tqdm(trainloader, desc=f"[{fusion_type}] Ep{epoch}/{args.max_epochs}")
        for batch in pbar:
            images = batch['image'].cuda()
            labels = batch['label'].cuda()

            seg_out, edge_pred = model(images)
            loss, ld = criterion(seg_out, edge_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            d = dice_score_batch(seg_out, labels, args.num_classes)
            ep_loss += loss.item(); ep_dice += d
            ep_edge += ld['edge']; ep_bnd += ld['boundary']; n += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{d:.4f}",
                'edge': f"{ld['edge']:.4f}",
                'bnd':  f"{ld['boundary']:.4f}",
            })

        scheduler.step()
        avg_loss = ep_loss/n; avg_dice = ep_dice/n
        print(f"\nEpoch {epoch}/{args.max_epochs} [{fusion_type}]")
        print(f"  Loss {avg_loss:.4f} | Dice {avg_dice:.4f} | Edge {ep_edge/n:.4f} | Boundary {ep_bnd/n:.4f}")
        print(f"  LR {optimizer.param_groups[0]['lr']:.2e}")

        # Latest checkpoint (for resume)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'loss': avg_loss,
            'fusion_type': fusion_type,
        }, os.path.join(out_dir, 'latest_checkpoint.pth'))

        # Best model
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_dice': best_dice,
                'loss': avg_loss,
                'fusion_type': fusion_type,
            }, os.path.join(out_dir, 'best_model.pth'))
            print(f"  ✓ Best model saved (Dice={best_dice:.4f})")

        if epoch % args.eval_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_dice': avg_dice,
                'loss': avg_loss,
            }, os.path.join(out_dir, f'epoch_{epoch}.pth'))

    print(f"\n✓ [{fusion_type}] Training done. Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',     default='./data/Synapse/train_npz')
    parser.add_argument('--list_dir',      default='./lists/lists_Synapse')
    parser.add_argument('--num_classes',   type=int,   default=9)
    parser.add_argument('--max_epochs',    type=int,   default=400)
    parser.add_argument('--batch_size',    type=int,   default=10)
    parser.add_argument('--base_lr',       type=float, default=1e-4)
    parser.add_argument('--img_size',      type=int,   default=224)
    parser.add_argument('--seed',          type=int,   default=1234)
    parser.add_argument('--output_dir',    default='./results_boundary_aware')
    parser.add_argument('--eval_interval', type=int,   default=20)
    parser.add_argument('--dataset',       default='synapse', choices=['synapse','cvc'])
    parser.add_argument('--img_dir',       default='./ds/img')
    parser.add_argument('--mask_dir',      default='./ds/mask')
    parser.add_argument('--fusion_types',  nargs='+',
                        default=['concat', 'weighted_sum'])
    args = parser.parse_args()

    cudnn.benchmark=False; cudnn.deterministic=True
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)

    for fusion_type in args.fusion_types:
        train_one(args, fusion_type)
