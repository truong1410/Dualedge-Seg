"""
Visualize segmentation results - Convert NIfTI to PNG
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from glob import glob
import SimpleITK as sitk
from matplotlib.colors import ListedColormap

def load_nifti(path):
    """Load NIfTI file"""
    img = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(img)
    return array

def create_colormap(n_classes=9):
    """Tạo colormap cho 9 classes"""
    colors = [
        [0, 0, 0],        # Background - Black
        [1, 0, 0],        # Class 1 - Red (Aorta)
        [0, 1, 0],        # Class 2 - Green (Gallbladder)
        [0, 0, 1],        # Class 3 - Blue (Left Kidney)
        [1, 1, 0],        # Class 4 - Yellow (Right Kidney)
        [1, 0, 1],        # Class 5 - Magenta (Liver)
        [0, 1, 1],        # Class 6 - Cyan (Pancreas)
        [1, 0.5, 0],      # Class 7 - Orange (Spleen)
        [0.5, 0, 1],      # Class 8 - Purple (Stomach)
    ]
    return ListedColormap(colors)

def visualize_case(pred_path, img_path, gt_path, output_dir, case_name, num_slices=10):
    """Visualize 1 case"""
    
    # Load data
    pred = load_nifti(pred_path)
    img = load_nifti(img_path)
    gt = load_nifti(gt_path)
    
    print(f"Processing {case_name}...")
    print(f"  Shape: {img.shape}")
    print(f"  Slices: {img.shape[0]}")
    
    # Chọn slices đều nhau
    total_slices = img.shape[0]
    slice_indices = np.linspace(5, total_slices-5, num_slices, dtype=int)
    
    # Create colormap
    cmap = create_colormap()
    
    # Visualize
    fig, axes = plt.subplots(num_slices, 3, figsize=(15, 5*num_slices))
    
    for i, slice_idx in enumerate(slice_indices):
        # Input image
        axes[i, 0].imshow(img[slice_idx], cmap='gray')
        axes[i, 0].set_title(f'Slice {slice_idx}: Input Image', fontsize=12)
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(img[slice_idx], cmap='gray', alpha=0.7)
        axes[i, 1].imshow(gt[slice_idx], cmap=cmap, alpha=0.5, vmin=0, vmax=8)
        axes[i, 1].set_title('Ground Truth Overlay', fontsize=12)
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(img[slice_idx], cmap='gray', alpha=0.7)
        axes[i, 2].imshow(pred[slice_idx], cmap=cmap, alpha=0.5, vmin=0, vmax=8)
        axes[i, 2].set_title('Prediction Overlay', fontsize=12)
        axes[i, 2].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Aorta'),
        Patch(facecolor='green', label='Gallbladder'),
        Patch(facecolor='blue', label='Left Kidney'),
        Patch(facecolor='yellow', label='Right Kidney'),
        Patch(facecolor='magenta', label='Liver'),
        Patch(facecolor='cyan', label='Pancreas'),
        Patch(facecolor='orange', label='Spleen'),
        Patch(facecolor='purple', label='Stomach'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=10)
    
    plt.suptitle(f'{case_name} - Segmentation Results', fontsize=16, y=0.995)
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{case_name}_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {save_path}")
    plt.close()

def visualize_all_cases(pred_dir, output_dir='./visualizations'):
    """Visualize tất cả cases"""
    
    # Tìm tất cả predictions
    pred_files = sorted(glob(os.path.join(pred_dir, '*_pred.nii.gz')))
    
    print(f"Found {len(pred_files)} cases")
    print("="*50)
    
    for pred_path in pred_files:
        # Extract case name
        case_name = os.path.basename(pred_path).replace('_pred.nii.gz', '')
        
        # Tìm files tương ứng
        img_path = pred_path.replace('_pred.nii.gz', '_img.nii.gz')
        gt_path = pred_path.replace('_pred.nii.gz', '_gt.nii.gz')
        
        if os.path.exists(img_path) and os.path.exists(gt_path):
            visualize_case(pred_path, img_path, gt_path, output_dir, case_name)
        else:
            print(f"⚠ Missing files for {case_name}")
    
    print("="*50)
    print(f"✓ All visualizations saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, 
                        default='./predictions_dexined/BEFUnet_DexiNed',
                        help='Directory containing predictions')
    parser.add_argument('--output_dir', type=str,
                        default='./visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_slices', type=int, default=8,
                        help='Number of slices to visualize per case')
    
    args = parser.parse_args()
    
    visualize_all_cases(args.pred_dir, args.output_dir)
