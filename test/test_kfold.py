"""
Test all K-fold models and aggregate results.
For each fusion type: average metrics across all folds.
Also runs on fixed test set for final comparison.
"""

import argparse
import os
import json
import sys
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import zoom
from medpy import metric
import SimpleITK as sitk
from skimage import measure
from skimage.morphology import binary_erosion as sk_erosion
from scipy.ndimage import distance_transform_edt

np.bool = bool; np.int = int; np.float = float
np.complex = complex; np.object = object; np.str = str

import configs.BEFUnet_DualEdge_configs as cfg_module


FIXED_TEST = [
    'case0001','case0002','case0003','case0004','case0008',
    'case0022','case0025','case0029','case0032','case0035',
    'case0036','case0038',
]

CLASS_NAMES = ['Aorta','Gallbladder','L.Kidney','R.Kidney',
               'Liver','Pancreas','Spleen','Stomach']


# ── Metric helpers ────────────────────────────────────────────────────────────

def euler_char(mask):
    if mask.sum() == 0: return 0
    mask = mask.astype(np.uint8)
    try: return int(measure.euler_number(mask, connectivity=2 if mask.ndim==2 else 3))
    except: return int(measure.label(mask,connectivity=1).max())

def euler_diff(p, g):
    p,g=(p>0).astype(np.uint8),(g>0).astype(np.uint8)
    if p.sum()==0 and g.sum()==0: return 0.
    return float(abs(euler_char(p)-euler_char(g)))

def calc_nsd(p, g, spacing=None, tol=2.0):
    if spacing is None: spacing=[1.,1.,1.]
    if p.sum()==0 or g.sum()==0: return 0.
    try:
        ps=p.astype(bool)^sk_erosion(p.astype(bool))
        gs=g.astype(bool)^sk_erosion(g.astype(bool))
        if ps.sum()==0 or gs.sum()==0: return 0.
        dp=distance_transform_edt(~ps,sampling=spacing)
        dg=distance_transform_edt(~gs,sampling=spacing)
        d=np.concatenate([dp[gs],dg[ps]])
        return float((d<=tol).sum()/len(d))
    except: return 0.

def metrics_pc(pred, gt, spacing=None):
    if spacing is None: spacing=[1.,1.,1.]
    pb=(pred>0).astype(np.uint8); gb=(gt>0).astype(np.uint8)
    if pb.sum()>0 and gb.sum()>0:
        d={'dice':metric.binary.dc(pb,gb),'ed':euler_diff(pb,gb),'nsd':calc_nsd(pb,gb,spacing)}
        try: d['hd95']=metric.binary.hd95(pb,gb,voxelspacing=spacing)
        except: d['hd95']=np.nan
        try: d['asd']=metric.binary.asd(pb,gb,voxelspacing=spacing)
        except: d['asd']=np.nan
    elif pb.sum()>0:
        d={'dice':0.,'hd95':np.nan,'ed':float(euler_char(pb)),'asd':np.nan,'nsd':0.}
    else:
        d={'dice':0.,'hd95':np.nan,'ed':0. if gb.sum()==0 else float(euler_char(gb)),'asd':np.nan,'nsd':0.}
    return d


# ── Test one volume ───────────────────────────────────────────────────────────

def test_volume(image, label, model, num_classes, img_size, z_spacing=1):
    """image, label: numpy [D, H, W]"""
    D, H, W = image.shape
    pred = np.zeros_like(label)
    model.eval()
    for si in range(D):
        s = image[si]
        if H != img_size or W != img_size:
            s = zoom(s, (img_size/H, img_size/W), order=3)
        x = torch.from_numpy(np.stack([s,s,s],0)).unsqueeze(0).float().cuda()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(model(x),1),1).squeeze(0).cpu().numpy()
            if H != img_size or W != img_size:
                out = zoom(out, (H/img_size, W/img_size), order=0)
            pred[si] = out
    sp = [float(z_spacing), 1., 1.]
    return [metrics_pc((pred==i),(label==i),sp) for i in range(1, num_classes)]


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(ckpt_path, fusion_type, num_classes, img_size):
    import models.dualedge_backbone as bb
    config = cfg_module.get_BEFUnet_DualEdge_configs()

    if fusion_type == 'weighted_sum':
        fusion_cls = bb.DualEdgeFusion
    elif fusion_type == 'cross_attention':
        fusion_cls = bb.CrossAttentionFusion
    else:
        fusion_cls = bb.ConcatFusion

    orig_init = bb.DualEdgeBackbone.__init__
    def patched_init(self, config, in_channels=3):
        orig_init(self, config, in_channels)
        self.fuse = fusion_cls(channels=config.cnn_pyramid_fm)
    bb.DualEdgeBackbone.__init__ = patched_init

    from models.BEFUnet_DualEdge import BEFUnet_DualEdge
    model = BEFUnet_DualEdge(config, img_size=img_size,
                              in_chans=3, n_classes=num_classes).cuda()
    bb.DualEdgeBackbone.__init__ = orig_init

    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold_dir',    default='./results_kfold')
    parser.add_argument('--h5_dir',       default='./data/Synapse/test_vol_h5')
    parser.add_argument('--num_classes',  type=int, default=9)
    parser.add_argument('--img_size',     type=int, default=224)
    parser.add_argument('--z_spacing',    type=int, default=1)
    parser.add_argument('--n_folds',      type=int, default=5)
    parser.add_argument('--fusion_types', nargs='+',
                        default=['weighted_sum','cross_attention','concat'])
    parser.add_argument('--is_savenii',   action='store_true')
    args = parser.parse_args()

    cudnn.benchmark=False; cudnn.deterministic=True

    log_dir = os.path.join(args.kfold_dir, 'test_results')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'test_kfold.txt'),
        level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    import h5py
    all_results = {}

    for fusion_type in args.fusion_types:
        logging.info(f"\n{'='*70}")
        logging.info(f"FUSION TYPE: {fusion_type}")
        logging.info('='*70)

        fold_test_metrics = []  # metrics per fold on fixed test set

        for fold_idx in range(args.n_folds):
            ckpt_path = os.path.join(
                args.kfold_dir, f'fusion_{fusion_type}',
                f'fold_{fold_idx}', 'best_model.pth')

            if not os.path.exists(ckpt_path):
                logging.info(f"Fold {fold_idx}: checkpoint not found, skipping")
                continue

            logging.info(f"\n--- Fold {fold_idx} ---")
            model = load_model(ckpt_path, fusion_type, args.num_classes, args.img_size)

            case_metrics = []
            for case in tqdm(FIXED_TEST, desc=f'{fusion_type} fold{fold_idx}'):
                h5_path = None
                for ext in ['.npy.h5', '.h5']:
                    p = os.path.join(args.h5_dir, case + ext)
                    if os.path.exists(p):
                        h5_path = p; break
                if h5_path is None:
                    logging.info(f"  {case}: h5 not found, skip")
                    continue

                with h5py.File(h5_path, 'r') as f:
                    image = f['image'][:]
                    label = f['label'][:]

                mi = test_volume(image, label, model,
                                 args.num_classes, args.img_size, args.z_spacing)
                case_metrics.append(mi)

                case_dice = np.mean([m['dice'] for m in mi])
                logging.info(f"  {case}: Dice={case_dice:.4f}")

            if not case_metrics:
                continue

            # Aggregate across cases for this fold
            nc = args.num_classes - 1
            arr = {k: np.array([[m[k] for m in cm] for cm in case_metrics])
                   for k in ['dice','hd95','ed','asd','nsd']}

            fold_summary = {}
            for i, cname in enumerate(CLASS_NAMES):
                fold_summary[cname] = {
                    'dice': float(np.mean(arr['dice'][:,i])),
                    'hd95': float(np.nanmean(arr['hd95'][:,i])),
                    'ed':   float(np.mean(arr['ed'][:,i])),
                    'asd':  float(np.nanmean(arr['asd'][:,i])),
                    'nsd':  float(np.mean(arr['nsd'][:,i])),
                }

            fold_mean_dice = float(np.mean(arr['dice']))
            fold_summary['mean_dice'] = fold_mean_dice
            fold_test_metrics.append(fold_summary)

            logging.info(f"  Fold {fold_idx} mean Dice: {fold_mean_dice:.4f}")

        # Average across folds
        if not fold_test_metrics:
            continue

        logging.info(f"\n=== {fusion_type} AVERAGED ACROSS {len(fold_test_metrics)} FOLDS ===")
        avg_summary = {}
        for cname in CLASS_NAMES:
            avg_summary[cname] = {}
            for k in ['dice','hd95','ed','asd','nsd']:
                vals = [f[cname][k] for f in fold_test_metrics]
                avg_summary[cname][k] = {
                    'mean': float(np.nanmean(vals)),
                    'std':  float(np.nanstd(vals)),
                }
            logging.info(
                f"  {cname:12s}: "
                f"Dice {avg_summary[cname]['dice']['mean']:.4f}±{avg_summary[cname]['dice']['std']:.4f}, "
                f"HD95 {avg_summary[cname]['hd95']['mean']:.2f}mm, "
                f"NSD {avg_summary[cname]['nsd']['mean']:.4f}"
            )

        mean_dices = [f['mean_dice'] for f in fold_test_metrics]
        logging.info(f"\n  OVERALL Mean Dice: {np.mean(mean_dices):.4f} ± {np.std(mean_dices):.4f}")
        avg_summary['overall_mean_dice'] = {
            'mean': float(np.mean(mean_dices)),
            'std':  float(np.std(mean_dices)),
        }
        all_results[fusion_type] = avg_summary

    # Save
    with open(os.path.join(log_dir, 'kfold_test_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Final comparison table
    logging.info("\n" + "="*70)
    logging.info("FINAL COMPARISON TABLE")
    logging.info("="*70)
    header = f"{'Fusion':20s} | " + " | ".join(f"{c[:8]:8s}" for c in CLASS_NAMES) + " | Mean"
    logging.info(header)
    logging.info("-"*len(header))
    for fusion_type, res in all_results.items():
        row = f"{fusion_type:20s} | "
        row += " | ".join(f"{res[c]['dice']['mean']*100:8.2f}" for c in CLASS_NAMES)
        row += f" | {res['overall_mean_dice']['mean']*100:.2f}"
        logging.info(row)

    logging.info(f"\n✓ Results saved to {log_dir}")
