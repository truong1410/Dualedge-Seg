"""Training script for TopologyGuidedNet on Synapse."""
import argparse, os, random, sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb

np.bool=bool; np.int=int; np.float=float
np.complex=complex; np.object=object; np.str=str

from models.TopologyGuidedNet import TopologyGuidedNet, TopologyGuidedLoss


def dice_score_batch(outputs, labels, n_classes):
    preds=torch.argmax(torch.softmax(outputs.detach(),dim=1),dim=1)
    dices=[]
    for c in range(1,n_classes):
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
    return {'image':torch.stack(images,0),
            'label':torch.stack([s['label'] for s in batch],0),
            'case_name':[s.get('case_name','') for s in batch]}

parser=argparse.ArgumentParser()
parser.add_argument('--root_path',     default='./data/Synapse/train_npz')
parser.add_argument('--list_dir',      default='./lists/lists_Synapse')
parser.add_argument('--test_path',     default='./data/Synapse/test_vol_h5')
parser.add_argument('--num_classes',   type=int,   default=9)
parser.add_argument('--max_epochs',    type=int,   default=400)
parser.add_argument('--batch_size',    type=int,   default=10)
parser.add_argument('--base_lr',       type=float, default=1e-4)
parser.add_argument('--img_size',      type=int,   default=224)
parser.add_argument('--lambda_topo',   type=float, default=0.1)
parser.add_argument('--n_thresholds',  type=int,   default=20)
parser.add_argument('--seed',          type=int,   default=1234)
parser.add_argument('--output_dir',    default='./results_topology_guided')
parser.add_argument('--model_name',    default='TopologyGuidedNet')
parser.add_argument('--eval_interval', type=int,   default=20)
parser.add_argument('--patience',      type=int,   default=50)
args=parser.parse_args()
args.output_dir=os.path.join(args.output_dir, args.model_name)
os.makedirs(args.output_dir, exist_ok=True)

if __name__=='__main__':
    cudnn.benchmark=False; cudnn.deterministic=True
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)

    wandb.init(project='BEFUnet-ablation', name=args.model_name,
               config={'model':'TopologyGuidedNet',
                       'lambda_topo':args.lambda_topo,
                       'n_thresholds':args.n_thresholds,
                       'max_epochs':args.max_epochs,
                       'batch_size':args.batch_size})

    model     = TopologyGuidedNet(n_classes=args.num_classes).cuda()
    criterion = TopologyGuidedLoss(
        n_classes=args.num_classes,
        lambda_topo=args.lambda_topo,
        n_thresholds=args.n_thresholds)

    n_params=sum(p.numel() for p in model.parameters())/1e6
    print(f"✓ TopologyGuidedNet  {n_params:.2f}M params")

    optimizer=optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6)

    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    db_train=Synapse_dataset(
        base_dir=args.root_path, list_dir=args.list_dir, split='train',
        transform=transforms.Compose([RandomGenerator([args.img_size]*2)]))
    trainloader=DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, collate_fn=collate_fn)

    db_val=Synapse_dataset(args.test_path, split='test_vol', list_dir=args.list_dir)
    from torch.utils.data import DataLoader as DL
    valloader=DL(db_val, batch_size=1, shuffle=False, num_workers=1)
    print(f"✓ Train: {len(db_train)}, Val: {len(db_val)}")

    best_dice=-1.0; start_epoch=1; no_improve=0

    resume_path=os.path.join(args.output_dir,'latest_checkpoint.pth')
    if os.path.exists(resume_path):
        ckpt=torch.load(resume_path,map_location='cuda',weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch=ckpt['epoch']+1; best_dice=ckpt.get('best_dice',-1.0)
        no_improve=ckpt.get('no_improve',0)
        print(f"✓ Resumed from epoch {ckpt['epoch']} (best_dice={best_dice:.4f})")

    for epoch in range(start_epoch, args.max_epochs+1):
        model.train()
        ep_loss=0; ep_dice=0; ep_topo=0; n=0

        pbar=tqdm(trainloader, desc=f"Ep{epoch}/{args.max_epochs}")
        for batch in pbar:
            images=batch['image'].cuda(); labels=batch['label'].cuda()
            logits=model(images)
            loss, ld=criterion(logits, labels)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            d=dice_score_batch(logits, labels, args.num_classes)
            ep_loss+=loss.item(); ep_dice+=d; ep_topo+=ld['topo']; n+=1
            pbar.set_postfix({'loss':f"{loss.item():.4f}",
                              'dice':f"{d:.4f}",'topo':f"{ld['topo']:.4f}"})

        scheduler.step()
        avg_loss=ep_loss/n; avg_dice=ep_dice/n
        print(f"\nEpoch {epoch}/{args.max_epochs} | Loss {avg_loss:.4f} | "
              f"Dice {avg_dice:.4f} | Topo {ep_topo/n:.4f} | "
              f"LR {optimizer.param_groups[0]['lr']:.2e}")

        wandb.log({'epoch':epoch,'loss':avg_loss,'dice':avg_dice,
                   'topo_loss':ep_topo/n,'lr':optimizer.param_groups[0]['lr']})

        torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'best_dice':best_dice,'loss':avg_loss,'no_improve':no_improve},
                   os.path.join(args.output_dir,'latest_checkpoint.pth'))

        # Validation
        val_dice=avg_dice
        if epoch % args.eval_interval == 0 or epoch == args.max_epochs:
            model.eval()
            val_dices=[]
            from scipy.ndimage import zoom
            for vbatch in valloader:
                vimg=vbatch['image'].squeeze(0).cpu().numpy()
                vlbl=vbatch['label'].squeeze(0).cpu().numpy()
                D,H,W=vimg.shape; vpred=np.zeros_like(vlbl)
                for si in range(D):
                    s=vimg[si]
                    if H!=args.img_size or W!=args.img_size:
                        s=zoom(s,(args.img_size/H,args.img_size/W),order=3)
                    vx=torch.from_numpy(np.stack([s,s,s],0)).unsqueeze(0).float().cuda()
                    with torch.no_grad():
                        vout=model(vx)
                        vp=torch.argmax(torch.softmax(vout,1),1).squeeze(0).cpu().numpy()
                        if H!=args.img_size or W!=args.img_size:
                            vp=zoom(vp,(H/args.img_size,W/args.img_size),order=0)
                        vpred[si]=vp
                for c in range(1,args.num_classes):
                    p=(vpred==c).astype(float); g=(vlbl==c).astype(float)
                    inter=(p*g).sum(); union=p.sum()+g.sum()
                    if union>0: val_dices.append(2*inter/union)
            val_dice=float(np.mean(val_dices)) if val_dices else 0.
            model.train()
            print(f"  Val Dice: {val_dice:.4f}")
            wandb.log({'val_dice':val_dice,'epoch':epoch})

        if val_dice>best_dice:
            best_dice=val_dice; no_improve=0
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),
                        'best_dice':best_dice,'loss':avg_loss},
                       os.path.join(args.output_dir,'best_model.pth'))
            print(f"  ✓ Best model saved (Val Dice={best_dice:.4f})")
            wandb.run.summary['best_dice']=best_dice
        else:
            no_improve+=1
            if no_improve>=args.patience:
                print(f"\n⚠ Early stopping at epoch {epoch}")
                wandb.finish(); break

        if epoch%args.eval_interval==0:
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),
                        'best_dice':avg_dice,'loss':avg_loss},
                       os.path.join(args.output_dir,f'epoch_{epoch}.pth'))

    print(f"\n✓ Done. Best Dice: {best_dice:.4f}")
    wandb.finish()
