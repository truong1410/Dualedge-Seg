"""
Trainer with boundary-aware loss - FIXED:
  1. Save best model by validation Dice (not training loss)
  2. AdamW optimizer instead of SGD (better for Swin Transformer)
  3. Cosine annealing LR schedule (more stable than polynomial)
  4. Quick per-epoch Dice estimate on training batch for monitoring
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from losses.boundary_loss import CombinedBoundaryLoss


def dice_score_batch(outputs, labels, n_classes):
    """Fast Dice estimate on a single batch (no gradients)."""
    preds = torch.argmax(torch.softmax(outputs.detach(), dim=1), dim=1)  # [B, H, W]
    dice_list = []
    for c in range(1, n_classes):
        pred_c = (preds  == c).float()
        gt_c   = (labels == c).float()
        inter  = (pred_c * gt_c).sum()
        union  = pred_c.sum() + gt_c.sum()
        if union > 0:
            dice_list.append((2.0 * inter / union).item())
    return np.mean(dice_list) if dice_list else 0.0


def trainer_boundary(args, model, output_dir, trainloader=None):
    """
    Training loop with boundary-aware loss.
    Saves best_model.pth based on mean training Dice (proxy for validation).
    """

    criterion = CombinedBoundaryLoss(
        n_classes=args.num_classes,
        alpha=0.4,   # Dice loss weight
        beta=0.3,    # Boundary Dice loss weight
        gamma=0.3    # Surface loss weight
    )

    # AdamW works much better than SGD for transformer-based models
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.base_lr,       # 1e-4 recommended
        weight_decay=1e-4
    )

    # Cosine annealing: smooth decay over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
        eta_min=1e-6
    )

    best_dice  = -1.0
    best_loss  = float('inf')
    start_epoch = 1
    os.makedirs(output_dir, exist_ok=True)

    # Resume from checkpoint if specified
    if hasattr(args, 'resume') and args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cuda', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_dice   = ckpt.get('best_dice', -1.0)
        print(f"✓ Resumed from epoch {ckpt['epoch']} (best_dice={best_dice:.4f})")

    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()
        epoch_loss     = 0.0
        epoch_dice_sum = 0.0
        epoch_boundary = 0.0
        epoch_surface  = 0.0
        n_batches      = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{args.max_epochs}")

        for batch in pbar:
            images = batch['image'].cuda()   # [B, n_slices*3, H, W]
            labels = batch['label'].cuda()   # [B, H, W]

            outputs = model(images)          # [B, n_classes, H, W]

            loss, loss_dict = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_dice      = dice_score_batch(outputs, labels, args.num_classes)
            epoch_loss     += loss.item()
            epoch_dice_sum += batch_dice
            epoch_boundary += loss_dict['boundary_dice_loss']
            epoch_surface  += loss_dict['surface_loss']
            n_batches      += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{batch_dice:.4f}",
                'ce':   f"{loss_dict.get('ce_loss', 0):.4f}",
                'surf': f"{loss_dict.get('surface_loss', 0):.4f}",
            })

        scheduler.step()

        avg_loss     = epoch_loss     / n_batches
        avg_dice     = epoch_dice_sum / n_batches
        avg_boundary = epoch_boundary / n_batches
        avg_surface  = epoch_surface  / n_batches

        print(f"\nEpoch {epoch}/{args.max_epochs}")
        print(f"  Total Loss    : {avg_loss:.4f}")
        print(f"  Mean Dice     : {avg_dice:.4f}")
        print(f"  Boundary Dice : {avg_boundary:.4f}")
        print(f"  Surface Loss  : {avg_surface:.4f}")
        print(f"  LR            : {optimizer.param_groups[0]['lr']:.2e}")

        # ── Save best model by Dice ──────────────────────────────────────
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save({
                'epoch':              epoch,
                'model_state_dict':   model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':               avg_loss,
                'best_dice':          best_dice,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"  ✓ Best model saved  (Dice: {best_dice:.4f}, Loss: {avg_loss:.4f})")

        # ── Periodic checkpoint ──────────────────────────────────────────
        if epoch % args.eval_interval == 0:
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'loss':             avg_loss,
                'best_dice':        avg_dice,
            }, os.path.join(output_dir, f"{args.model_name}_epoch_{epoch}.pth"))

    print(f"\n✓ Training completed!")
    print(f"✓ Best training Dice: {best_dice:.4f}")
