"""Test script for BoundaryAwareNetV2 on Synapse."""
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

from models.BoundaryAwareNetV2 import BoundaryAwareNetV2
import configs.boundary_aware_configs as cfg_module

CLASS_NAMES=['Aorta','Gallbladder','L.Kidney','R.Kidney','Liver','Pancreas','Spleen','Stomach']

def euler_char(mask):
    if mask.sum()==0: return 0
    try: return int(measure.euler_number(mask.astype(np.uint8),connectivity=2 if mask.ndim==2 else 3))
    except: return int(measure.label(mask,connectivity=1).max())

def calc_nsd(p,g,spacing=None,tol=2.0):
    if spacing is None: spacing=[1.,1.,1.]
    if p.sum()==0 or g.sum()==0: return 0.
    try:
        ps=p.astype(bool)^sk_erosion(p.astype(bool)); gs=g.astype(bool)^sk_erosion(g.astype(bool))
        if ps.sum()==0 or gs.sum()==0: return 0.
        dp=distance_transform_edt(~ps,sampling=spacing); dg=distance_transform_edt(~gs,sampling=spacing)
        d=np.concatenate([dp[gs],dg[ps]]); return float((d<=tol).sum()/len(d))
    except: return 0.

def metrics_pc(pred,gt,spacing=None):
    if spacing is None: spacing=[1.,1.,1.]
    pb=(pred>0).astype(np.uint8); gb=(gt>0).astype(np.uint8)
    if pb.sum()>0 and gb.sum()>0:
        d={'dice':metric.binary.dc(pb,gb),'ed':float(abs(euler_char(pb)-euler_char(gb))),'nsd':calc_nsd(pb,gb,spacing)}
        try: d['hd95']=metric.binary.hd95(pb,gb,voxelspacing=spacing)
        except: d['hd95']=np.nan
        try: d['asd']=metric.binary.asd(pb,gb,voxelspacing=spacing)
        except: d['asd']=np.nan
    elif pb.sum()>0:
        d={'dice':0.,'hd95':np.nan,'ed':float(euler_char(pb)),'asd':np.nan,'nsd':0.}
    else:
        d={'dice':0.,'hd95':np.nan,'ed':0. if gb.sum()==0 else float(euler_char(gb)),'asd':np.nan,'nsd':0.}
    return d

def test_volume(image,label,model,num_classes,img_size,z_spacing=1,save_path=None,case=None):
    D,H,W=image.shape; pred=np.zeros_like(label); model.eval()
    for si in range(D):
        s=image[si]
        if H!=img_size or W!=img_size: s=zoom(s,(img_size/H,img_size/W),order=3)
        x=torch.from_numpy(np.stack([s,s,s],0)).unsqueeze(0).float().cuda()
        with torch.no_grad():
            seg_out,_=model(x)
            p=torch.argmax(torch.softmax(seg_out,1),1).squeeze(0).cpu().numpy()
            if H!=img_size or W!=img_size: p=zoom(p,(H/img_size,W/img_size),order=0)
            pred[si]=p
    sp=[float(z_spacing),1.,1.]
    mlist=[metrics_pc((pred==i),(label==i),sp) for i in range(1,num_classes)]
    if save_path and case:
        for arr,tag in [(image,'img'),(pred,'pred'),(label,'gt')]:
            itk=sitk.GetImageFromArray(arr.astype(np.float32))
            itk.SetSpacing((1,1,float(z_spacing)))
            sitk.WriteImage(itk,os.path.join(save_path,f"{case}_{tag}.nii.gz"))
    return mlist

def inference(args,model,save_path=None):
    from datasets.dataset_synapse import Synapse_dataset
    db=Synapse_dataset(args.test_path,split='test_vol',list_dir=args.list_dir)
    loader=DataLoader(db,batch_size=1,shuffle=False,num_workers=1)
    logging.info(f"Test cases: {len(loader)}"); model.eval()
    all_m={k:[] for k in ['dice','hd95','ed','asd','nsd']}; case_m={}
    for batch in tqdm(loader,total=len(loader)):
        name=batch['case_name'][0]
        img=batch['image'].squeeze(0).cpu().numpy()
        lbl=batch['label'].squeeze(0).cpu().numpy()
        mi=test_volume(img,lbl,model,args.num_classes,args.img_size,args.z_spacing,save_path,name)
        case_m[name]={k:np.nanmean([m[k] for m in mi]) for k in all_m}
        logging.info('Case '+name+': '+', '.join(f"{k}={case_m[name][k]:.4f}" for k in all_m))
        for m in mi:
            for k in all_m: all_m[k].append(m[k])
    nc=args.num_classes-1; ncs=len(loader)
    arr={k:np.array(all_m[k]).reshape(ncs,nc) for k in all_m}
    logging.info("\n"+"="*80+"\nPER-CLASS RESULTS:\n"+"="*80)
    for i,cname in enumerate(CLASS_NAMES):
        logging.info(f"Class {i+1} ({cname:12s}): Dice {np.mean(arr['dice'][:,i]):.4f}, HD95 {np.nanmean(arr['hd95'][:,i]):.2f}mm, ED {np.mean(arr['ed'][:,i]):.2f}, ASD {np.nanmean(arr['asd'][:,i]):.2f}mm, NSD {np.mean(arr['nsd'][:,i]):.4f}")
    means={k:(np.mean if k not in ('hd95','asd') else np.nanmean)(arr[k]) for k in all_m}
    logging.info("\n"+"="*80+"\nOVERALL PERFORMANCE:\n"+"="*80)
    for k,v in means.items(): logging.info(f"  Mean {k.upper():5s}: {v:.4f}")
    logging.info("="*80)
    return means,case_m

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--test_path',    default='./data/Synapse/test_vol_h5')
    parser.add_argument('--list_dir',     default='./lists/lists_Synapse')
    parser.add_argument('--model_weight', required=True)
    parser.add_argument('--fusion_type',  default='weighted_sum',
                        choices=['weighted_sum','concat','cross_attention'])
    parser.add_argument('--num_classes',  type=int, default=9)
    parser.add_argument('--img_size',     type=int, default=224)
    parser.add_argument('--z_spacing',    type=int, default=1)
    parser.add_argument('--output_dir',   default='./predictions_boundary_aware_v2')
    parser.add_argument('--model_name',   default='BoundaryAwareNetV2')
    parser.add_argument('--is_savenii',   action='store_true')
    parser.add_argument('--seed',         type=int, default=1234)
    args=parser.parse_args()

    cudnn.benchmark=False; cudnn.deterministic=True
    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)

    config=cfg_module.get_boundary_aware_configs()
    model=BoundaryAwareNetV2(config,n_classes=args.num_classes,
                              fusion_type=args.fusion_type,img_size=args.img_size).cuda()
    ckpt=torch.load(args.model_weight,map_location='cuda',weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    print("✓ Model loaded\n")

    log_dir='./test_log/test_log_boundary_aware_v2_v2'; os.makedirs(log_dir,exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir,args.model_name+'.txt'),
        level=logging.INFO,format='[%(asctime)s] %(message)s',datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    save_path=None
    if args.is_savenii:
        save_path=os.path.join(args.output_dir,args.model_name); os.makedirs(save_path,exist_ok=True)

    means,case_m=inference(args,model,save_path)
    import pandas as pd
    pd.DataFrame.from_dict(case_m,orient='index').to_csv(
        os.path.join(log_dir,f'{args.model_name}_results.csv'))
    print(f"\n✓ Done!")
