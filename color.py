import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
nii_dir = "./predictions/BEFUnet"
out_dir = "./png_output_color"
alpha_pred = 0.45
alpha_gt = 0.25
save_gt = True

os.makedirs(out_dir, exist_ok=True)

# =========================
# FIXED COLOR MAP (RGB)
# =========================
CLASS_COLORS = {
    0: (0, 0, 0),         # background
    1: (255, 0, 0),       # spleen
    2: (0, 255, 0),       # right kidney
    3: (0, 0, 255),       # left kidney
    4: (255, 255, 0),     # gallbladder
    5: (0, 255, 255),     # esophagus
    6: (255, 165, 0),     # liver
    7: (128, 0, 128),     # stomach
    8: (255, 105, 180),   # aorta
}


# =========================
# HELPER
# =========================
def normalize_ct(img):
    img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def label_to_rgb(mask):
    """Convert label map → RGB image"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[mask == cls] = color
    return rgb


# =========================
# MAIN
# =========================
cases = sorted(set(
    f.replace("_img.nii.gz", "")
    for f in os.listdir(nii_dir)
    if f.endswith("_img.nii.gz")
))

for case in cases:
    print(f"[INFO] Processing {case}")

    img = nib.load(os.path.join(nii_dir, case + "_img.nii.gz")).get_fdata()
    pred = nib.load(os.path.join(nii_dir, case + "_pred.nii.gz")).get_fdata().astype(np.int32)

    gt_path = os.path.join(nii_dir, case + "_gt.nii.gz")
    gt = nib.load(gt_path).get_fdata().astype(np.int32) if save_gt and os.path.exists(gt_path) else None

    img = normalize_ct(img)

    case_out = os.path.join(out_dir, case)
    os.makedirs(case_out, exist_ok=True)

    for z in range(img.shape[2]):
        ct_slice = img[:, :, z]
        pred_rgb = label_to_rgb(pred[:, :, z])

        plt.figure(figsize=(4, 4))
        plt.imshow(ct_slice, cmap="gray")

        plt.imshow(pred_rgb, alpha=alpha_pred)

        if gt is not None:
            gt_rgb = label_to_rgb(gt[:, :, z])
            plt.imshow(gt_rgb, alpha=alpha_gt)

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(case_out, f"slice_{z:03d}.png"), dpi=200)
        plt.close()

print("✅ DONE: Fixed-color PNG exported.")
