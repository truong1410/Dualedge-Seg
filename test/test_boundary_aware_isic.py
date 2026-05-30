"""Test script for BoundaryAwareNet — metrics: Dice, HD95, ED, ASD, NSD."""

import argparse, logging, os, sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

np.bool=bool; np.int=int; np.float=float
np.complex=complex; np.object=object; np.str=str

from scipy.ndimage import zoom, distance_transform_edt
from medpy import metric
import SimpleITK as sitk
from skimage import measure
from skimage.morphology import binary_erosion as sk_erosion

from models.boundary_aware_net import BoundaryAwareNet
import configs.boundary_aware_configs as cfg_module

CLASS_NAMES = ['Aorta','Gallbladder','L.Kidney','R.Kidney',
               'Liver','Pancreas','Spleen','Stomach']


def euler_char(mask):
    if mask.sum()==0: return 0
    mask=mask.astype(np.uint8)
    try: return int(measure.euler_number(mask, connectivity=2 if mask.ndim==2 else 3))
    except: return int(measure.label(mask,connectivity=1).max())

def euler_diff(p,g):
    p,g=(p>0).astype(np.uint8),(g>0).astype(np.uint8)
    if p.sum()==0 and g.sum()==0: return 0.
    return float(abs(euler_char(p)-euler_char(g)))

def calc_nsd(p,g,spacing=None,tol=2.0):
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

def metrics_pc(pred,gt,spacing=None):
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

    D,H,W=image.shape; pred=np.zeros_like(label)
    model.eval()
    for si in range(D):
        s=image[si]
        if H!=img_size or W!=img_size: s=zoom(s,(img_size/H,img_size/W),order=3)
        x=torch.from_numpy(np.stack([s,s,s],0)).unsqueeze(0).float().cuda()
        with torch.no_grad():
            seg_out, _ = model(x)
            out=torch.argmax(torch.softmax(seg_out,1),1).squeeze(0).cpu().numpy()
            if H!=img_size or W!=img_size: out=zoom(out,(H/img_size,W/img_size),order=0)
            pred[si]=out
    sp=[float(z_spacing),1.,1.]
    mlist=[metrics_pc((pred==i),(label==i),sp) for i in range(1,num_classes)]
    if save_path and case:
        for arr,tag in [(image,'img'),(pred,'pred'),(label,'gt')]:
            itk=sitk.GetImageFromArray(arr.astype(np.float32))
            itk.SetSpacing((1,1,float(z_spacing)))
            sitk.WriteImage(itk,os.path.join(save_path,f"{case}_{tag}.nii.gz"))
    return mlist


def metrics_binary(pred, gt):
    from skimage.morphology import binary_erosion as sk_erosion
    from scipy.ndimage import distance_transform_edt
    pb=(pred>0).astype(np.uint8); gb=(gt>0).astype(np.uint8)
    if pb.sum()>0 and gb.sum()>0:
        d={"dice": metric.binary.dc(pb,gb)}
        try: d["hd95"]=metric.binary.hd95(pb,gb)
        except: d["hd95"]=np.nan
        try: d["asd"]=metric.binary.asd(pb,gb)
        except: d["asd"]=np.nan
        try:
            ps=pb.astype(bool)^sk_erosion(pb.astype(bool))
            gs=gb.astype(bool)^sk_erosion(gb.astype(bool))
            dp=distance_transform_edt(~ps); dg=distance_transform_edt(~gs)
            dists=np.concatenate([dp[gs],dg[ps]])
            d["nsd"]=float((dists<=2.0).sum()/len(dists))
        except: d["nsd"]=0.
        inter=(pb*gb).sum(); union=pb.sum()+gb.sum()
        d["iou"]=float(inter/(union-inter+1e-8))
    else:
        d={"dice":0.,"hd95":np.nan,"asd":np.nan,"nsd":0.,"iou":0.}
    return d

def inference(args, model, save_path=None):
    from datasets.dataset_isic import ISIC_dataset, RandomGenerator_ISIC
    db     = ISIC_dataset(args.img_dir, args.mask_dir, args.list_dir, split="test",
                         transform=RandomGenerator_ISIC([args.img_size]*2))
    loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"Test cases: {len(loader)}")
    model.eval()
    all_m  = {k:[] for k in ["dice","hd95","asd","nsd","iou"]}
    case_m = {}
    for batch in tqdm(loader, total=len(loader)):
        image = batch["image"].cuda()
        label = batch["label"].squeeze(0).cpu().numpy()
        name  = batch["case_name"][0]
        with torch.no_grad():
            out, _ = model(image)
            pred = torch.argmax(torch.softmax(out,1),1).squeeze(0).cpu().numpy()
        m = metrics_binary(pred, label)
        for k in all_m: all_m[k].append(m[k])
        logging.info(f"{name}: dice={m['dice']:.4f}, iou={m['iou']:.4f}, hd95={m['hd95']:.2f}, nsd={m['nsd']:.4f}")
    logging.info("\n"+"="*60+"\nOVERALL PERFORMANCE\n"+"="*60)
    means = {}
    for k,v in all_m.items():
        fn = np.mean if k not in ("hd95","asd") else np.nanmean
        means[k] = fn(v)
        logging.info(f"  Mean {k.upper():5s}: {means[k]:.4f}")
    return means, case_m


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--list_dir',     default='./lists/lists_Synapse')
    parser.add_argument('--model_weight', required=True)
    parser.add_argument('--fusion_type',  default='weighted_sum',
                        choices=['concat','weighted_sum','cross_attention'])
    parser.add_argument('--num_classes',  type=int, default=9)
    parser.add_argument('--img_size',     type=int, default=224)
    parser.add_argument('--z_spacing',    type=int, default=1)
    parser.add_argument('--output_dir',   default='./predictions_boundary_aware')
    parser.add_argument('--model_name',   default='BoundaryAwareNet')
    parser.add_argument('--img_dir',      default='./ds/img')
    parser.add_argument('--mask_dir',     default='./ds/mask')
    parser.add_argument('--is_savenii',   action='store_true')
    parser.add_argument('--seed',         type=int, default=1234)
    args=parser.parse_args()

    cudnn.benchmark=False; cudnn.deterministic=True
    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)

    config=cfg_module.get_boundary_aware_configs()
    # Detect fusion type from checkpoint if not specified
    ckpt_check = torch.load(args.model_weight, map_location='cpu', weights_only=False)
    if 'fusion_type' in ckpt_check:
        detected = ckpt_check['fusion_type']
        if detected != args.fusion_type:
            print(f'⚠ fusion_type mismatch: arg={args.fusion_type}, checkpoint={detected}')
            print(f'  Using checkpoint fusion_type: {detected}')
            args.fusion_type = detected
    print(f"Loading model [{args.fusion_type}] from {args.model_weight}")
    model=BoundaryAwareNet(config,n_classes=args.num_classes,
                           fusion_type=args.fusion_type,img_size=args.img_size).cuda()
    ckpt=torch.load(args.model_weight,map_location='cuda',weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    print("✓ Model loaded\n")

    log_dir='./test_log/test_log_boundary_aware_isic'; os.makedirs(log_dir,exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir,args.model_name+'.txt'),
        level=logging.INFO,format='[%(asctime)s] %(message)s',datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    save_path=None
    if args.is_savenii:
        save_path=os.path.join(args.output_dir,args.model_name)
        os.makedirs(save_path,exist_ok=True)

    means,case_m=inference(args,model,save_path)
    import pandas as pd
    pd.DataFrame.from_dict(case_m,orient='index').to_csv(
        os.path.join(log_dir,f'{args.model_name}_results.csv'))
    print(f"\n✓ Results saved to {log_dir}\n✓ Done!")
