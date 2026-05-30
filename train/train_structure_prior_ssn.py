"""Training script for StructurePriorSSN on Synapse."""
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

from models.StructurePriorSSN import StructurePriorSSN, UncertaintyGuidedLoss


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
parser.add_argument('--batch_size',    type=int,   default=8)
parser.add_argument('--base_lr',       type=float, default=1e-4)
parser.add_argument('--img_size',      type=int,   default=224)
parser.add_argument('--embed_dim',     type=int,   default=256)
parser.add_argument('--depth',         type=int,   default=6)
parser.add_argument('--d_state',       type=int,   default=16)
parser.add_argument('--prior_dim',     type=int,   default=64)
parser.add_argument('--seed',          type=int,   default=1234)
parser.add_argument('--output_dir',    default='./results_structure_prior_ssn')
parser.add_argument('--model_name',    default='StructurePriorSSN')
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
               config={'model':'StructurePriorSSN','embed_dim':args.embed_dim,
                       'depth':args.depth,'d_state':args.d_state,
                       'prior_dim':args.prior_dim,
                       'max_epochs':args.max_epochs,'batch_size':args.batch_size})

    model=StructurePriorSSN(
        n_classes=args.num_classes, img_size=args.img_size,
        embed_dim=args.embed_dim, depth=args.depth,
        prior_dim=args.prior_dim, d_state=args.d_state).cuda()

    n_params=sum(p.numel() for p in model.parameters())/1e6
    print(f"✓ StructurePriorSSN  {n_params:.2f}M params")

    criterion=UncertaintyGuidedLoss(n_classes=args.num_classes)
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
        ep_loss=0; ep_dice=0; ep_unc=0; n=0

        pbar=tqdm(trainloader, desc=f"Ep{epoch}/{args.max_epochs}")
        for batch in pbar:
            images=batch['image'].cuda(); labels=batch['label'].cuda()
            # Pass GT label for structure prior during training
            seg_out, unc_out=model(images, label=labels)
            loss, ld=criterion(seg_out, unc_out, labels)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            d=dice_score_batch(seg_out, labels, args.num_classes)
            ep_loss+=loss.item(); ep_dice+=d; ep_unc+=ld['unc']; n+=1
            pbar.set_postfix({'loss':f"{loss.item():.4f}",
                              'dice':f"{d:.4f}",'unc':f"{ld['unc']:.4f}"})

        scheduler.step()
        avg_loss=ep_loss/n; avg_dice=ep_dice/n
        print(f"\nEpoch {epoch}/{args.max_epochs} | Loss {avg_loss:.4f} | "
              f"Dice {avg_dice:.4f} | Unc {ep_unc/n:.4f} | "
              f"LR {optimizer.param_groups[0]['lr']:.2e}")

        wandb.log({'epoch':epoch,'loss':avg_loss,'dice':avg_dice,
                   'unc_loss':ep_unc/n,'lr':optimizer.param_groups[0]['lr']})

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
                        vout,_=model(vx)  # no label at inference
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
