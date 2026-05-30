"""Training script for BEFUnet on CVC-ClinicDB (binary polyp segmentation)."""

import argparse, os, random, sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

np.bool=bool; np.int=int; np.float=float
np.complex=complex; np.object=object; np.str=str

from datasets.dataset_isic import ISIC_dataset, RandomGenerator_ISIC
from losses.boundary_loss import CombinedBoundaryLoss


def dice_score_batch(outputs, labels, n_classes):
    preds = torch.argmax(torch.softmax(outputs.detach(), dim=1), dim=1)
    dices = []
    for c in range(1, n_classes):
        p = (preds==c).float(); g = (labels==c).float()
        inter = (p*g).sum(); union = p.sum()+g.sum()
        if union > 0: dices.append((2*inter/union).item())
    return np.mean(dices) if dices else 0.0


parser = argparse.ArgumentParser()
parser.add_argument("--img_dir",       default="./ds/img")
parser.add_argument("--mask_dir",      default="./ds/mask")
parser.add_argument("--list_dir",      default="./lists/lists_CVC")
parser.add_argument("--num_classes",   type=int,   default=2)
parser.add_argument("--max_epochs",    type=int,   default=200)
parser.add_argument("--batch_size",    type=int,   default=16)
parser.add_argument("--base_lr",       type=float, default=1e-4)
parser.add_argument("--img_size",      type=int,   default=224)
parser.add_argument("--seed",          type=int,   default=1234)
parser.add_argument("--output_dir",    default="./results_isic")
parser.add_argument("--model_name",    default="BEFUnet_CVC")
parser.add_argument("--eval_interval", type=int,   default=20)
parser.add_argument("--fusion_type",   default="weighted_sum", choices=["weighted_sum","concat","cross_attention"])
parser.add_argument("--model_type",    default="dexined",
                    choices=["dexined", "dualedge"],
                    help="dexined=BEFUnet+DexiNed, dualedge=BEFUnet+DualEdge")
args = parser.parse_args()
args.output_dir = os.path.join(args.output_dir, args.model_name)
os.makedirs(args.output_dir, exist_ok=True)


if __name__ == "__main__":
    cudnn.benchmark=False; cudnn.deterministic=True
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)

    # Build model
    if args.model_type == "dexined":
        import configs.BEFUnet_DexiNed_2_5D_configs as cfg_module
        from models.BEFUnet_DexiNed_2_5D import BEFUnet_DexiNed_2_5D
        config = cfg_module.get_BEFUnet_DexiNed_2_5D_configs(n_slices=1)
        model  = BEFUnet_DexiNed_2_5D(
            config=config, img_size=args.img_size,
            in_chans=3, n_classes=args.num_classes, n_slices=1).cuda()
    else:
        import configs.BEFUnet_DualEdge_configs as cfg_module
        from models.BEFUnet_DualEdge import BEFUnet_DualEdge
        import models.dualedge_backbone as bb
        fusion_map = {"weighted_sum": bb.DualEdgeFusion.__class__, "concat": bb.ConcatFusion, "cross_attention": bb.CrossAttentionFusion}
        # Patch fusion
        orig = bb.DualEdgeBackbone.__init__
        fusion_cls = {"weighted_sum": bb.DualEdgeFusion, "concat": bb.ConcatFusion, "cross_attention": bb.CrossAttentionFusion}[args.fusion_type]
        def _patched(self, config, in_channels=3):
            orig(self, config, in_channels)
            self.fuse = fusion_cls(channels=config.cnn_pyramid_fm)
        bb.DualEdgeBackbone.__init__ = _patched
        config = cfg_module.get_BEFUnet_DualEdge_configs()
        model  = BEFUnet_DualEdge(config=config, img_size=args.img_size,
            in_chans=3, n_classes=args.num_classes).cuda()
        bb.DualEdgeBackbone.__init__ = orig

    n_params = sum(p.numel() for p in model.parameters())/1e6
    print(f"✓ Model: {args.model_name} ({args.model_type}) — {n_params:.2f}M params")

    # Reset class weights for binary segmentation (2 classes)
    from losses.boundary_loss import CombinedBoundaryLoss as _CBL
    import torch.nn as nn
    criterion = _CBL(n_classes=args.num_classes, alpha=0.3, beta=0.2, gamma=0.1, w_ce=0.4)
    # Override CE with no class weights for binary
    criterion.ce = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6)

    db_train = ISIC_dataset(
        img_dir=args.img_dir, mask_dir=args.mask_dir,
        list_dir=args.list_dir, split="train",
        transform=RandomGenerator_ISIC([args.img_size]*2))
    trainloader = DataLoader(db_train, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    print(f"✓ Training samples: {len(db_train)}")

    best_dice   = -1.0
    start_epoch = 1

    resume_path = os.path.join(args.output_dir, "latest_checkpoint.pth")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cuda", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", -1.0)
        print(f"✓ Resumed from epoch {ckpt['epoch']} (best_dice={best_dice:.4f})")

    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()
        ep_loss=0; ep_dice=0; n=0

        pbar = tqdm(trainloader, desc=f"Ep{epoch}/{args.max_epochs}")
        for batch in pbar:
            images = batch["image"].cuda()
            labels = batch["label"].cuda()
            outputs = model(images)
            loss, ld = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            d = dice_score_batch(outputs, labels, args.num_classes)
            ep_loss+=loss.item(); ep_dice+=d; n+=1
            pbar.set_postfix({"loss":f"{loss.item():.4f}","dice":f"{d:.4f}"})

        scheduler.step()
        avg_loss=ep_loss/n; avg_dice=ep_dice/n
        print(f"\nEpoch {epoch}/{args.max_epochs} | Loss {avg_loss:.4f} | Dice {avg_dice:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "loss": avg_loss,
        }, os.path.join(args.output_dir, "latest_checkpoint.pth"))

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_dice": best_dice,
                "loss": avg_loss,
            }, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  ✓ Best model saved (Dice={best_dice:.4f})")

        if epoch % args.eval_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_dice": avg_dice,
                "loss": avg_loss,
            }, os.path.join(args.output_dir, f"epoch_{epoch}.pth"))

    print(f"\n✓ Training done. Best Dice: {best_dice:.4f}")
