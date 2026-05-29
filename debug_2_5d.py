"""
Debug 2.5D with fixes
"""

import torch
from torchvision import transforms
from datasets.dataset_synapse_2_5d import Synapse_dataset_2_5D, RandomGenerator_2_5D
from models.BEFUnet_DexiNed_2_5D import BEFUnet_DexiNed_2_5D
import configs.BEFUnet_DexiNed_2_5D_configs as configs

print("="*60)
print("TESTING 2.5D IMPLEMENTATION")
print("="*60)

# 1. Test dataset
print("\n1. Testing Dataset...")
dataset = Synapse_dataset_2_5D(
    base_dir='./data/Synapse/train_npz',
    list_dir='./lists/lists_Synapse',
    split='train',
    n_slices=3,
    transform=transforms.Compose([
        RandomGenerator_2_5D(output_size=[224, 224])
    ])
)

sample = dataset[0]
print(f"   Image shape: {sample['image'].shape}")  # Should be [9, 224, 224]
print(f"   Label shape: {sample['label'].shape}")  # Should be [224, 224]

expected_channels = 9  # 3 slices × 3 RGB
actual_channels = sample['image'].shape[0]

if actual_channels == expected_channels:
    print(f"   ✓ Correct! Image has {actual_channels} channels (3 slices × 3 RGB)")
else:
    print(f"   ✗ ERROR! Expected {expected_channels} channels, got {actual_channels}")
    print(f"   → Dataset transform needs fixing")

# 2. Test model
print("\n2. Testing Model...")
config = configs.get_BEFUnet_DexiNed_2_5D_configs(n_slices=3)
model = BEFUnet_DexiNed_2_5D(
    config=config,
    img_size=224,
    in_chans=9,
    n_classes=9,
    n_slices=3
).cuda()

print(f"   Model expects: 9 channels (3 slices × 3 RGB)")
print(f"   ✓ Model created successfully")

# 3. Test with dummy data
print("\n3. Testing Forward Pass (Dummy Data)...")
dummy_input = torch.randn(2, 9, 224, 224).cuda()  # [B, 9, H, W]
print(f"   Input shape: {dummy_input.shape}")

try:
    output = model(dummy_input)
    print(f"   ✓ Forward pass successful!")
    print(f"   Output shape: {output.shape}")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")

# 4. Test with real data
print("\n4. Testing Forward Pass (Real Data)...")
real_input = sample['image'].unsqueeze(0).cuda()  # Add batch dimension
print(f"   Input shape: {real_input.shape}")

try:
    output = model(real_input)
    print(f"   ✓ Real data forward pass successful!")
    print(f"   Output shape: {output.shape}")
    
    # Check output
    pred = torch.argmax(output, dim=1)
    print(f"   Prediction shape: {pred.shape}")
    print(f"   Prediction range: [{pred.min()}, {pred.max()}]")
    
except Exception as e:
    print(f"   ✗ Real data forward pass failed: {e}")

# 5. Test batch loading
print("\n5. Testing Batch DataLoader...")
from torch.utils.data import DataLoader

def collate_fn(batch):
    images = torch.stack([s['image'] for s in batch])
    labels = torch.stack([s['label'] for s in batch])
    return {'image': images, 'label': labels}

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

try:
    batch = next(iter(loader))
    print(f"   Batch image shape: {batch['image'].shape}")  # Should be [4, 9, 224, 224]
    print(f"   Batch label shape: {batch['label'].shape}")  # Should be [4, 224, 224]
    
    if batch['image'].shape[1] == 9:
        print(f"   ✓ DataLoader working correctly!")
    else:
        print(f"   ✗ DataLoader issue: expected 9 channels, got {batch['image'].shape[1]}")
        
except Exception as e:
    print(f"   ✗ DataLoader failed: {e}")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"Dataset output channels: {actual_channels}")
print(f"Model expected channels: {expected_channels}")
print(f"Match: {'✓ YES' if actual_channels == expected_channels else '✗ NO'}")
print("="*60)
