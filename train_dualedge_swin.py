"""
Training script for DualEdge_Swin experiments.
Covers 4 experiments:
  A: DualEdge_UNet  + CE+Dice only        (no boundary loss)
  B: DualEdge_UNet  + full boundary loss  (already = BoundaryAwareNet)
  C: DualEdge_Swin  + CE+Dice only
  D: DualEdge_Swin  + full boundary loss
"""

import argparse, os, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb

np.bool=bool; np.int=int; np.float=float
np.complex=complex; np.object=object; np.str=str

from models.DualEdge_Swin import DualEdge_Swin
import configs.BEFUnet_DualEdge_configs as cfg_module


# ── Simple CE+Dice loss (no boundary terms) ──────────────────────────────────

class SimpleLoss(nn.Module):
    def __init__(self, n_classes, smooth=1e-5):
        super().__init__()
        self.ce     = nn.CrossEntropyLoss()
        self.n_cl   = n_classes
        self.smooth = smooth

    def forward(self, logits, target):
        ce   = self.ce(logits, target)
        prob = torch.softmax(logits, dim=1)
        toh  = torch.nn.functional.one_hot(target, self.n_cl).permute(0,3,1,2).float()
        B,C,H,W = prob.shape
        p    = prob.view(B,C,-1); t = toh.view(B,C,-1)
        inter= (p*t).sum(2); union = p.sum(2)+t.sum(2)
        dice = (2*inter+self.smooth)/(union+self.smooth)
        return ce + (1-dice[:,1:].mean()).mean(), {'ce':ce.item(),'dice':dice[:,1:].mean().item()}


# ── Boundary-aware loss (reuse existing) ─────────────────────────────────────

def get_loss(loss_type, n_classes):
    if loss_type == 'simple':
        return SimpleLoss(n_classes)
    else:
        from losses.boundary_aware_loss import BoundaryAwareLoss
        return BoundaryAwareLoss(n_classes=n_classes, lambda1=0.3, lambda2=0.2)


def dice_score_batch(outputs, labels, n_classes):
    preds = torch.argmax(torch.softmax(outputs.detach(), dim=1), dim=1)
    dices = []
    for c in range(1, n_classes):
        p=(preds==c).float(); g=(labels==c).float()
        inter=(p*g).sum(); union=p.sum()+g.sum()
        if union>0: dices.append((2*inter/union).item())
    return np.mean(dices) if dices else 0.0


def collate_fn(batch):
    images=[]
    for s in batch:
        img=s['image']
        if img.shape[0]==1: img=img.repeat(3,1,1)
        images.append(img)
    return {'image': torch.stack(images,0),
            'label': torch.stack([s['label'] for s in batch],0),
            'case_name': [s.get('case_name','') for s in batch]}


parser = argparse.ArgumentParser()
parser.add_argument('--root_path',     default='./data/Synapse/train_npz')
parser.add_argument('--list_dir',      default='./lists/lists_Synapse')
parser.add_argument('--num_classes',   type=int,   default=9)
parser.add_argument('--max_epochs',    type=int,   default=400)
parser.add_argument('--batch_size',    type=int,   default=10)
parser.add_argument('--base_lr',       type=float, default=1e-4)
parser.add_argument('--img_size',      type=int,   default=224)
parser.add_argument('--seed',          type=int,   default=1234)
parser.add_argument('--output_dir',    default='./results_ablation')
parser.add_argument('--eval_interval', type=int,   default=20)
parser.add_argument('--fusion_type',   default='weighted_sum',
                    choices=['weighted_sum','concat','cross_attention'])
parser.add_argument('--patience',      type=int,   default=50, help='Stop if no improvement for N epochs')
parser.add_argument('--loss_type',     default='simple',
                    choices=['simple','boundary'],
                    help='simple=CE+Dice only, boundary=full boundary loss')
args = parser.parse_args()

exp_name = f"DualEdge_Swin_{args.fusion_type}_{'boundary' if args.loss_type=='boundary' else 'nodloss'}"
args.output_dir = os.path.join(args.output_dir, exp_name)
os.makedirs(args.output_dir, exist_ok=True)

if __name__ != '__main__':
    pass
else:
    wandb.init(
        project='BEFUnet-ablation',
        name=exp_name,
        config={
            'fusion_type': args.fusion_type,
            'loss_type':   args.loss_type,
            'max_epochs':  args.max_epochs,
            'batch_size':  args.batch_size,
            'base_lr':     args.base_lr,
            'model':       'DualEdge_Swin',
        }
    )


if __name__ == '__main__':
    cudnn.benchmark=False; cudnn.deterministic=True
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)

    config = cfg_module.get_BEFUnet_DualEdge_configs()
    model  = DualEdge_Swin(
        config, n_classes=args.num_classes,
        fusion_type=args.fusion_type, img_size=args.img_size).cuda()

    n_params = sum(p.numel() for p in model.parameters())/1e6
    print(f"✓ DualEdge_Swin [{args.fusion_type}] [{args.loss_type}]  {n_params:.2f}M params")

    criterion = get_loss(args.loss_type, args.num_classes)
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

    best_dice=-1.0; start_epoch=1; no_improve=0

    resume_path = os.path.join(args.output_dir, 'latest_checkpoint.pth')
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location='cuda', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice   = ckpt.get('best_dice', -1.0)
        print(f"✓ Resumed from epoch {ckpt['epoch']} (best_dice={best_dice:.4f})")

    for epoch in range(start_epoch, args.max_epochs+1):
        model.train()
        ep_loss=0; ep_dice=0; n=0

        pbar = tqdm(trainloader, desc=f"[{exp_name}] Ep{epoch}/{args.max_epochs}")
        for batch in pbar:
            images=batch['image'].cuda(); labels=batch['label'].cuda()
            seg_out, edge_out = model(images)

            if args.loss_type == 'simple':
                loss, ld = criterion(seg_out, labels)
            else:
                loss, ld = criterion(seg_out, edge_out, labels)
            outputs = seg_out

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            d=dice_score_batch(outputs, labels, args.num_classes)
            ep_loss+=loss.item(); ep_dice+=d; n+=1
            pbar.set_postfix({'loss':f"{loss.item():.4f}",'dice':f"{d:.4f}"})

        scheduler.step()
        avg_loss=ep_loss/n; avg_dice=ep_dice/n
        print(f"\nEpoch {epoch}/{args.max_epochs} | Loss {avg_loss:.4f} | Dice {avg_dice:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}")
        wandb.log({
            'epoch':     epoch,
            'loss':      avg_loss,
            'dice':      avg_dice,
            'lr':        optimizer.param_groups[0]['lr'],
        })

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice, 'loss': avg_loss,
            'fusion_type': args.fusion_type, 'loss_type': args.loss_type,
        }, os.path.join(args.output_dir, 'latest_checkpoint.pth'))

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'best_dice': best_dice, 'loss': avg_loss,
                'fusion_type': args.fusion_type, 'loss_type': args.loss_type,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  ✓ Best model saved (Dice={best_dice:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n⚠ Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                wandb.finish()
                break
            wandb.run.summary['best_dice'] = best_dice

        if epoch % args.eval_interval == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'best_dice': avg_dice, 'loss': avg_loss},
                       os.path.join(args.output_dir, f'epoch_{epoch}.pth'))

    print(f"\n✓ Done. Best Dice: {best_dice:.4f}")
    wandb.finish()
