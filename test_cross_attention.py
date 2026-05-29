"""Test script for BoundaryAwareNet with Cross Attention."""

import argparse, logging, os, sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import zoom
from medpy import metric
import SimpleITK as sitk

sys.path.append('.')
from models.boundary_aware_net_cross_attention import BoundaryAwareNetWithCrossAttention
import configs.boundary_aware_configs as cfg_module

CLASS_NAMES = ['Aorta','Gallbladder','L.Kidney','R.Kidney',
               'Liver','Pancreas','Spleen','Stomach']

def test_volume(image, label, model, num_classes, img_size, z_spacing=1, save_path=None, case=None):
    D, H, W = image.shape
    pred = np.zeros_like(label)
    model.eval()
    
    # Pad for 2.5D
    pad_slices = 1
    image_padded = np.pad(image, ((pad_slices, pad_slices), (0, 0), (0, 0)), mode='edge')
    
    for si in range(D):
        # Get 3 slices
        s = np.stack([
            image_padded[si],
            image_padded[si+1],
            image_padded[si+2]
        ], axis=0)
        
        # Resize if needed
        if H != img_size or W != img_size:
            s_resized = np.zeros((3, img_size, img_size))
            for c in range(3):
                s_resized[c] = zoom(s[c], (img_size/H, img_size/W), order=3)
            s = s_resized
        
        x = torch.from_numpy(s).unsqueeze(0).float().cuda()
        with torch.no_grad():
            seg_out, edge_out, attn_weights = model(x)
            out = torch.argmax(torch.softmax(seg_out, 1), 1).squeeze(0).cpu().numpy()
            
            if H != img_size or W != img_size:
                out = zoom(out, (H/img_size, W/img_size), order=0)
            pred[si] = out
    
    # Calculate metrics
    spacing = [float(z_spacing), 1., 1.]
    metrics_list = []
    for i in range(1, num_classes):
        pred_class = (pred == i).astype(np.uint8)
        gt_class = (label == i).astype(np.uint8)
        
        if pred_class.sum() > 0 and gt_class.sum() > 0:
            dice = metric.binary.dc(pred_class, gt_class)
            try:
                hd95 = metric.binary.hd95(pred_class, gt_class, voxelspacing=spacing)
            except:
                hd95 = np.nan
            try:
                asd = metric.binary.asd(pred_class, gt_class, voxelspacing=spacing)
            except:
                asd = np.nan
            metrics_list.append({'dice': dice, 'hd95': hd95, 'asd': asd})
        else:
            metrics_list.append({'dice': 0., 'hd95': np.nan, 'asd': np.nan})
    
    # Save predictions
    if save_path and case:
        for arr, tag in [(image, 'img'), (pred, 'pred'), (label, 'gt')]:
            itk = sitk.GetImageFromArray(arr.astype(np.float32))
            itk.SetSpacing((1, 1, float(z_spacing)))
            sitk.WriteImage(itk, os.path.join(save_path, f"{case}_{tag}.nii.gz"))
    
    return metrics_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='./data/Synapse/test_vol_h5')
    parser.add_argument('--list_dir', default='./lists/lists_Synapse')
    parser.add_argument('--model_weight', required=True)
    parser.add_argument('--fusion_type', default='concat')
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--z_spacing', type=int, default=1)
    parser.add_argument('--output_dir', default='./predictions_cross_attention')
    parser.add_argument('--is_savenii', action='store_true')
    args = parser.parse_args()
    
    # Setup
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(1234)
    
    # Load model
    config = cfg_module.get_boundary_aware_configs()
    print(f"Loading Cross Attention model from {args.model_weight}")
    
    model = BoundaryAwareNetWithCrossAttention(
        config, 
        n_classes=args.num_classes,
        fusion_type=args.fusion_type, 
        img_size=args.img_size
    ).cuda()
    
    ckpt = torch.load(args.model_weight, map_location='cuda')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded epoch {ckpt.get('epoch', 'unknown')}, best dice: {ckpt.get('best_dice', 'unknown')}")
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    print("✓ Model loaded\n")
    
    # Load data
    from datasets.dataset_synapse import Synapse_dataset
    dataset = Synapse_dataset(args.test_path, split='test_vol', list_dir=args.list_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Test
    all_dice = []
    save_path = args.output_dir if args.is_savenii else None
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    for batch in tqdm(loader, desc="Testing"):
        name = batch['case_name'][0]
        img = batch['image'].squeeze(0).cpu().numpy()
        lbl = batch['label'].squeeze(0).cpu().numpy()
        
        metrics = test_volume(img, lbl, model, args.num_classes, 
                             args.img_size, args.z_spacing, save_path, name)
        
        dice_scores = [m['dice'] for m in metrics]
        mean_dice = np.mean(dice_scores)
        all_dice.extend(dice_scores)
        print(f"{name}: Dice={mean_dice:.4f}")
    
    print(f"\n{'='*50}")
    print(f"Overall Mean Dice: {np.mean(all_dice):.4f}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
