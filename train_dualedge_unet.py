"""Training script for DualEdge_UNet on Synapse."""

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

from models.DualEdge_UNet import DualEdge_UNet
from losses.boundary_loss import CombinedBoundaryLoss
import configs.BEFUnet_DualEdge_configs as cfg_module


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
parser.add_argument('--output_dir',    default='./results_dualedge_unet')
parser.add_argument('--model_name',    default='DualEdge_UNet')
parser.add_argument('--eval_interval', type=int,   default=20)
parser.add_argument('--fusion_type',   default='weighted_sum',
                    choices=['weighted_sum','concat','cross_attention'])
args = parser.parse_args()
args.output_dir = os.path.join(args.output_dir, f"{args.model_name}_{args.fusion_type}")
os.makedirs(args.output_dir, exist_ok=True)


if __name__ == '__main__':
    cudnn.benchmark=False; cudnn.deterministic=True
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)

    config = cfg_module.get_BEFUnet_DualEdge_configs()
    model  = DualEdge_UNet(
        config, n_classes=args.num_classes,
        fusion_type=args.fusion_type, img_size=args.img_size).cuda()

    n_params = sum(p.numel() for p in model.parameters())/1e6
    print(f"✓ DualEdge_UNet [{args.fusion_type}]  {n_params:.2f}M params")

    criterion = CombinedBoundaryLoss(n_classes=args.num_classes)
    # Override CE with no class weights (clean for multi-class)
    import torch.nn as nn
    criterion.ce = nn.CrossEntropyLoss()

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

    best_dice=-1.0; start_epoch=1

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

        pbar = tqdm(trainloader, desc=f"[{args.fusion_type}] Ep{epoch}/{args.max_epochs}")
        for batch in pbar:
            images=batch['image'].cuda(); labels=batch['label'].cuda()
            outputs=model(images)
            loss, ld = criterion(outputs, labels)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            d=dice_score_batch(outputs, labels, args.num_classes)
            ep_loss+=loss.item(); ep_dice+=d; n+=1
            pbar.set_postfix({'loss':f"{loss.item():.4f}",'dice':f"{d:.4f}"})

        scheduler.step()
        avg_loss=ep_loss/n; avg_dice=ep_dice/n
        print(f"\nEpoch {epoch}/{args.max_epochs} [{args.fusion_type}]")
        print(f"  Loss {avg_loss:.4f} | Dice {avg_dice:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice, 'loss': avg_loss,
            'fusion_type': args.fusion_type,
        }, os.path.join(args.output_dir, 'latest_checkpoint.pth'))

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'best_dice': best_dice, 'loss': avg_loss,
                'fusion_type': args.fusion_type,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  ✓ Best model saved (Dice={best_dice:.4f})")

        if epoch % args.eval_interval == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'best_dice': avg_dice, 'loss': avg_loss},
                       os.path.join(args.output_dir, f'epoch_{epoch}.pth'))

    print(f"\n✓ Training done. Best Dice: {best_dice:.4f}")
