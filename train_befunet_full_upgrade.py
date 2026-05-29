"""
Full upgrade with FIXED collate function
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.BEFUnet_DexiNed_2_5D import BEFUnet_DexiNed_2_5D
from datasets.dataset_synapse_2_5d import Synapse_dataset_2_5D, RandomGenerator_2_5D
import configs.BEFUnet_DexiNed_2_5D_configs as configs
from trainer_boundary import trainer_boundary


# ========== ADD: Custom collate function ==========
def collate_fn_2_5d(batch):
    """
    Custom collate for 2.5D data
    
    Args:
        batch: List of samples, each with:
            - image: [n_slices, H, W]
            - label: [H, W]
    
    Returns:
        - images: [B, n_slices, H, W]
        - labels: [B, H, W]
    """
    images = []
    labels = []
    case_names = []
    
    for sample in batch:
        images.append(sample['image'])  # [n_slices, H, W]
        labels.append(sample['label'])  # [H, W]
        case_names.append(sample.get('case_name', ''))
    
    # Stack batch
    images = torch.stack(images, dim=0)  # [B, n_slices, H, W]
    labels = torch.stack(labels, dim=0)  # [B, H, W]
    
    return {
        'image': images,
        'label': labels,
        'case_name': case_names
    }
# ==================================================


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/Synapse/train_npz')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse')
parser.add_argument('--num_classes', type=int, default=9)
parser.add_argument('--max_epochs', type=int, default=401)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--n_slices', type=int, default=3)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--output_dir', type=str, default='./results_full_upgrade')
parser.add_argument('--model_name', type=str, default='BEFUnet_Full_Upgrade')
parser.add_argument('--eval_interval', type=int, default=20)

args = parser.parse_args()
args.output_dir = args.output_dir + f'/{args.model_name}'
os.makedirs(args.output_dir, exist_ok=True)


if __name__ == "__main__":
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Config
    config = configs.get_BEFUnet_DexiNed_2_5D_configs(n_slices=args.n_slices)
    in_channels = 3 * args.n_slices  # 9 for 3 slices
    
    # Model
    model = BEFUnet_DexiNed_2_5D(
        config=config,
        img_size=args.img_size,
        in_chans=in_channels,
        n_classes=args.num_classes,
        n_slices=args.n_slices
    ).cuda()
    
    print(f"✓ Model: {args.model_name}")
    print(f"✓ 2.5D input: {args.n_slices} slices → {in_channels} channels")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Dataset
    db_train = Synapse_dataset_2_5D(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split='train',
        n_slices=args.n_slices,
        transform=transforms.Compose([
            RandomGenerator_2_5D(output_size=[args.img_size, args.img_size])
        ])
    )
    
    print(f"✓ Training samples: {len(db_train)}")
    
    # ========== USE CUSTOM COLLATE ==========
    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_2_5d  # CRITICAL!
    )
    # ======================================
    
    # Test one batch
    print("\n Testing dataloader...")
    for batch in trainloader:
        images = batch['image']
        labels = batch['label']
        print(f"  Batch images shape: {images.shape}")  # Should be [B, n_slices, H, W]
        print(f"  Batch labels shape: {labels.shape}")  # Should be [B, H, W]
        print(f"  ✓ Dataloader working correctly!")
        break
    
    # Train
    trainer_boundary(args, model, args.output_dir, trainloader)
