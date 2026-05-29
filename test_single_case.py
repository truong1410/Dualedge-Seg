"""
Test script cho 1 case cụ thể và visualize kết quả
"""

import argparse
import os
import numpy as np
import torch
from scipy.ndimage import zoom
import h5py
import matplotlib.pyplot as plt
import cv2

from models.BEFUnet_DexiNed import BEFUnet_DexiNed
import configs.BEFUnet_DexiNed_configs as configs


def test_and_visualize(model, case_path, img_size=224, num_classes=9, save_dir='./visualizations'):
    """Test 1 case và visualize kết quả"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    h5f = h5py.File(case_path, 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    h5f.close()
    
    case_name = os.path.basename(case_path).replace('.npy.h5', '')
    print(f"Testing case: {case_name}")
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    
    model.eval()
    predictions = []
    
    # Process each slice
    for slice_idx in range(image.shape[0]):
        slice_img = image[slice_idx, :, :]
        x, y = slice_img.shape[0], slice_img.shape[1]
        
        # Resize if needed
        if x != img_size or y != img_size:
            slice_img = zoom(slice_img, (img_size / x, img_size / y), order=3)
        
        # Inference
        input_tensor = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float().cuda()
        
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
            pred = pred.cpu().numpy()
            
            # Resize back
            if x != img_size or y != img_size:
                pred = zoom(pred, (x / img_size, y / img_size), order=0)
        
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Visualize results
    num_slices = min(10, image.shape[0])  # Visualize tối đa 10 slices
    slice_indices = np.linspace(0, image.shape[0]-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(num_slices, 3, figsize=(15, 5*num_slices))
    
    for i, slice_idx in enumerate(slice_indices):
        # Original image
        axes[i, 0].imshow(image[slice_idx], cmap='gray')
        axes[i, 0].set_title(f'Slice {slice_idx}: Input Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(label[slice_idx], cmap='tab10', vmin=0, vmax=num_classes-1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(predictions[slice_idx], cmap='tab10', vmin=0, vmax=num_classes-1)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{case_name}_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.close()
    
    # Calculate Dice scores per class
    print(f"\n{'='*50}")
    print("Dice Scores per Class:")
    print(f"{'='*50}")
    
    for class_id in range(1, num_classes):
        pred_mask = (predictions == class_id).astype(np.float32)
        gt_mask = (label == class_id).astype(np.float32)
        
        intersection = np.sum(pred_mask * gt_mask)
        dice = (2. * intersection) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-8)
        
        print(f"Class {class_id}: Dice = {dice:.4f}")
    
    # Overall Dice
    pred_binary = (predictions > 0).astype(np.float32)
    gt_binary = (label > 0).astype(np.float32)
    intersection = np.sum(pred_binary * gt_binary)
    overall_dice = (2. * intersection) / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-8)
    
    print(f"\nOverall Dice Score: {overall_dice:.4f}")
    print(f"{'='*50}\n")
    
    return predictions, overall_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--case_path', type=str, required=True,
                        help='Path to test case (h5 file)')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--save_dir', type=str, default='./visualizations')
    
    args = parser.parse_args()
    
    # Load config and model
    config = configs.get_BEFUnet_DexiNed_configs()
    
    print("Loading model...")
    model = BEFUnet_DexiNed(
        config=config,
        img_size=args.img_size,
        n_classes=args.num_classes
    ).cuda()
    
    checkpoint = torch.load(args.checkpoint)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("✓ Model loaded!\n")
    
    # Test and visualize
    predictions, dice_score = test_and_visualize(
        model, args.case_path, args.img_size, args.num_classes, args.save_dir
    )
    
    print(f"✓ Testing completed!")
