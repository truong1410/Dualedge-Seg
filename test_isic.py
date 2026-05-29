"""Test script for BEFUnet on CVC-ClinicDB."""

import argparse, logging, os, sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import zoom, distance_transform_edt
from medpy import metric
from skimage.morphology import binary_erosion as sk_erosion

np.bool=bool; np.int=int; np.float=float
np.complex=complex; np.object=object; np.str=str

from datasets.dataset_isic import ISIC_dataset, RandomGenerator_ISIC


def calc_nsd(p, g, tol=2.0):
    if p.sum()==0 or g.sum()==0: return 0.
    try:
        ps=p.astype(bool)^sk_erosion(p.astype(bool))
        gs=g.astype(bool)^sk_erosion(g.astype(bool))
        if ps.sum()==0 or gs.sum()==0: return 0.
        dp=distance_transform_edt(~ps); dg=distance_transform_edt(~gs)
        d=np.concatenate([dp[gs],dg[ps]])
        return float((d<=tol).sum()/len(d))
    except: return 0.

def metrics_binary(pred, gt):
    pb=(pred>0).astype(np.uint8); gb=(gt>0).astype(np.uint8)
    if pb.sum()>0 and gb.sum()>0:
        d={"dice": metric.binary.dc(pb,gb), "nsd": calc_nsd(pb,gb)}
        try: d["hd95"]=metric.binary.hd95(pb,gb)
        except: d["hd95"]=np.nan
        try: d["asd"]=metric.binary.asd(pb,gb)
        except: d["asd"]=np.nan
        inter=(pb*gb).sum(); union=pb.sum()+gb.sum()
        d["iou"] = float(inter/(union-inter+1e-8))
    else:
        d={"dice":0.,"hd95":np.nan,"asd":np.nan,"nsd":0.,"iou":0.}
    return d


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--img_dir",      default="./ds/img")
    parser.add_argument("--mask_dir",     default="./ds/mask")
    parser.add_argument("--list_dir",     default="./lists/lists_CVC")
    parser.add_argument("--model_weight", required=True)
    parser.add_argument("--fusion_type",  default="cross_attention", choices=["weighted_sum","concat","cross_attention"])
    parser.add_argument("--model_type",   default="dexined",
                        choices=["dexined","dualedge"])
    parser.add_argument("--num_classes",  type=int, default=2)
    parser.add_argument("--img_size",     type=int, default=224)
    parser.add_argument("--model_name",   default="BEFUnet_CVC")
    parser.add_argument("--seed",         type=int, default=1234)
    args=parser.parse_args()

    cudnn.benchmark=False; cudnn.deterministic=True
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.model_type=="dexined":
        import configs.BEFUnet_DexiNed_2_5D_configs as cfg_module
        from models.BEFUnet_DexiNed_2_5D import BEFUnet_DexiNed_2_5D
        config=cfg_module.get_BEFUnet_DexiNed_2_5D_configs(n_slices=1)
        model=BEFUnet_DexiNed_2_5D(config=config,img_size=args.img_size,
                                    in_chans=3,n_classes=args.num_classes,n_slices=1).cuda()
    else:
        import configs.BEFUnet_DualEdge_configs as cfg_module
        from models.BEFUnet_DualEdge import BEFUnet_DualEdge
        import models.dualedge_backbone as bb
        fusion_cls = {"weighted_sum": bb.DualEdgeFusion, "concat": bb.ConcatFusion, "cross_attention": bb.CrossAttentionFusion}.get(args.fusion_type, bb.CrossAttentionFusion)
        orig = bb.DualEdgeBackbone.__init__
        def _patched(self, config, in_channels=3):
            orig(self, config, in_channels)
            self.fuse = fusion_cls(channels=config.cnn_pyramid_fm)
        bb.DualEdgeBackbone.__init__ = _patched
        config=cfg_module.get_BEFUnet_DualEdge_configs()
        model=BEFUnet_DualEdge(config=config,img_size=args.img_size,
                               in_chans=3,n_classes=args.num_classes).cuda()
        bb.DualEdgeBackbone.__init__ = orig

    ckpt=torch.load(args.model_weight,map_location="cuda",weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    print("✓ Model loaded")

    db_test=ISIC_dataset(args.img_dir,args.mask_dir,args.list_dir,split="test",
                       transform=RandomGenerator_ISIC([args.img_size]*2))
    loader=DataLoader(db_test,batch_size=1,shuffle=False,num_workers=1)

    log_dir="./test_log/test_log_isic"; os.makedirs(log_dir,exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir,args.model_name+".txt"),
        level=logging.INFO,format="[%(asctime)s] %(message)s",datefmt="%H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    model.eval()
    all_m={k:[] for k in ["dice","hd95","asd","nsd","iou"]}

    for batch in tqdm(loader,total=len(loader)):
        image  = batch["image"].cuda()   # [1,3,H,W]
        label  = batch["label"].squeeze(0).cpu().numpy()  # [H,W]
        name   = batch["case_name"][0]

        with torch.no_grad():
            out  = model(image)
            pred = torch.argmax(torch.softmax(out,1),1).squeeze(0).cpu().numpy()

        m=metrics_binary(pred,label)
        for k in all_m: all_m[k].append(m[k])
        logging.info(f"{name}: dice={m['dice']:.4f}, iou={m['iou']:.4f}, hd95={m['hd95']:.2f}, nsd={m['nsd']:.4f}")

    logging.info("\n"+"="*60+"\nOVERALL PERFORMANCE\n"+"="*60)
    for k,v in all_m.items():
        fn = np.mean if k not in ("hd95","asd") else np.nanmean
        logging.info(f"  Mean {k.upper():5s}: {fn(v):.4f}")

    import pandas as pd
    pd.DataFrame(all_m).to_csv(os.path.join(log_dir,f"{args.model_name}_results.csv"))
    print(f"\n✓ Results saved to {log_dir}")
