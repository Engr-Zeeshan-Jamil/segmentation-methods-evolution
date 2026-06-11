import os
import time
import random
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# =========================
# CONFIGURATION
# =========================

DATA_ROOT = "./data"
RESULTS_DIR = "./results_final"

NUM_IMAGES = 100
IMAGE_SIZE = 512
RANDOM_SEED = 42

NUM_CLASSES = 21
IGNORE_INDEX = 255

USE_WATERSHED_BINARY = True
USE_WATERSHED_SEMANTIC = True
USE_UNET_SEMANTIC = True
USE_SEGFORMER_SEMANTIC = True
USE_SAM_CLASS_AGNOSTIC = True

# U-Net semantic checkpoint
UNET_SEMANTIC_CHECKPOINT = "./unet_voc_semantic_21class.pth"
AUTO_TRAIN_UNET_IF_MISSING = True
UNET_TRAIN_IMAGE_SIZE = 256
UNET_TRAIN_EPOCHS = 8
UNET_TRAIN_BATCH_SIZE = 4
UNET_TRAIN_LR = 1e-4
UNET_TRAIN_MAX_IMAGES = 1000

# IMPORTANT:
# For correct Pascal VOC semantic comparison, use a SegFormer checkpoint trained on Pascal VOC.
# ADE20K has different label IDs, so the semantic result may not be fully valid for VOC classes.
SEGFORMER_CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"

# SAM
SAM_CHECKPOINT = "./sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/visualizations", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/charts", exist_ok=True)


# =========================
# BASIC UTILITIES
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_voc_dataset(image_set="val"):
    return VOCSegmentation(
        root=DATA_ROOT,
        year="2012",
        image_set=image_set,
        download=False
    )


def preprocess_image_pil(image_pil, size=512):
    image_pil = image_pil.convert("RGB")
    image_resized = image_pil.resize((size, size), Image.BILINEAR)
    return np.array(image_resized)


def preprocess_mask_pil(mask_pil, size=512):
    mask_np = np.array(mask_pil)
    mask_resized = Image.fromarray(mask_np).resize((size, size), Image.NEAREST)
    return np.array(mask_resized)


def convert_voc_to_binary(mask):
    """
    Pascal VOC:
    0 = background
    1-20 = object classes
    255 = ignore / boundary

    Output:
    0 = background
    1 = foreground
    255 = ignore
    """

    binary = np.zeros_like(mask, dtype=np.uint8)
    binary[(mask > 0) & (mask != IGNORE_INDEX)] = 1
    binary[mask == IGNORE_INDEX] = IGNORE_INDEX

    return binary


def measure_time(func, *args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    output = func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    return output, end_time - start_time


# =========================
# METRICS
# =========================

def compute_binary_metrics(pred, gt):
    valid_mask = gt != IGNORE_INDEX

    pred = pred[valid_mask]
    gt = gt[valid_mask]

    if len(gt) == 0:
        return 0.0, 0.0

    pixel_accuracy = np.mean(pred == gt)

    ious = []

    for cls in [0, 1]:
        pred_cls = pred == cls
        gt_cls = gt == cls

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            continue

        ious.append(intersection / union)

    miou = np.mean(ious) if len(ious) > 0 else 0.0

    return float(miou), float(pixel_accuracy)


def compute_semantic_metrics(pred, gt, num_classes=21):
    valid_mask = gt != IGNORE_INDEX

    pred = pred[valid_mask]
    gt = gt[valid_mask]

    if len(gt) == 0:
        return 0.0, 0.0

    pixel_accuracy = np.mean(pred == gt)

    ious = []

    for cls in range(num_classes):
        pred_cls = pred == cls
        gt_cls = gt == cls

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            continue

        ious.append(intersection / union)

    miou = np.mean(ious) if len(ious) > 0 else 0.0

    return float(miou), float(pixel_accuracy)


# =========================
# WATERSHED
# =========================

def watershed_markers(image_np):
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(
        dist_transform,
        0.4 * dist_transform.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image_bgr, markers)

    return markers


def watershed_binary_predict(image_np):
    markers = watershed_markers(image_np)

    pred = np.zeros(markers.shape, dtype=np.uint8)
    pred[markers > 1] = 1

    return pred

def watershed_semantic_oracle_predict(image_np, gt_semantic, foreground_ratio_threshold=0.05):
    """
    Improved oracle-assisted semantic Watershed.

    Watershed creates regions.
    Each region is assigned a Pascal VOC semantic class using ground truth.

    This is not a real deployable semantic model because it uses ground-truth
    labels during evaluation. It is only for oracle-style comparison.

    Logic:
    - If a Watershed region mostly overlaps background, assign background.
    - If it overlaps enough foreground, assign the majority foreground class.
    """

    markers = watershed_markers(image_np)

    pred_semantic = np.zeros(markers.shape, dtype=np.uint8)

    region_ids = np.unique(markers)
    region_ids = region_ids[region_ids > 1]

    for region_id in region_ids:
        region_mask = markers == region_id

        gt_values = gt_semantic[region_mask]

        # Remove ignore pixels
        gt_values = gt_values[gt_values != IGNORE_INDEX]

        if len(gt_values) == 0:
            pred_semantic[region_mask] = 0
            continue

        # Foreground means Pascal VOC class 1-20
        foreground_values = gt_values[(gt_values > 0) & (gt_values < IGNORE_INDEX)]

        foreground_ratio = len(foreground_values) / len(gt_values)

        # If region has very little object overlap, keep it background
        if foreground_ratio < foreground_ratio_threshold:
            pred_semantic[region_mask] = 0
            continue

        # Assign majority foreground class only
        values, counts = np.unique(foreground_values, return_counts=True)
        majority_foreground_class = values[np.argmax(counts)]

        pred_semantic[region_mask] = majority_foreground_class

    return pred_semantic

# =========================
# U-NET SEMANTIC DATASET
# =========================

class VOCSemanticSegmentationDataset(Dataset):
    def __init__(self, root, image_set="train", image_size=256, max_images=None):
        self.dataset = VOCSegmentation(
            root=root,
            year="2012",
            image_set=image_set,
            download=False
        )

        indices = list(range(len(self.dataset)))

        if max_images is not None:
            indices = indices[:max_images]

        self.indices = indices
        self.image_size = image_size

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_pil, mask_pil = self.dataset[self.indices[idx]]

        image_pil = image_pil.convert("RGB")
        image_tensor = self.image_transform(image_pil)

        mask_np = np.array(mask_pil)
        mask_resized = Image.fromarray(mask_np).resize(
            (self.image_size, self.image_size),
            Image.NEAREST
        )

        mask_np = np.array(mask_resized).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor


# =========================
# U-NET SEMANTIC
# =========================

def create_unet_semantic_model():
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )

    model.to(DEVICE)
    return model


def train_unet_semantic_checkpoint():
    print("\nU-Net semantic checkpoint not found.")
    print("Training U-Net Semantic on Pascal VOC train split...\n")

    train_dataset = VOCSemanticSegmentationDataset(
        root=DATA_ROOT,
        image_set="train",
        image_size=UNET_TRAIN_IMAGE_SIZE,
        max_images=UNET_TRAIN_MAX_IMAGES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=UNET_TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = create_unet_semantic_model()
    model.train()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=UNET_TRAIN_LR)

    for epoch in range(UNET_TRAIN_EPOCHS):
        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Training U-Net Semantic Epoch {epoch + 1}/{UNET_TRAIN_EPOCHS}"
        )

        for images, masks in progress:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1}/{UNET_TRAIN_EPOCHS} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), UNET_SEMANTIC_CHECKPOINT)

    print(f"\nSaved U-Net semantic checkpoint to: {UNET_SEMANTIC_CHECKPOINT}\n")

    return model


def load_unet_semantic_model():
    model = create_unet_semantic_model()

    if os.path.exists(UNET_SEMANTIC_CHECKPOINT):
        print(f"Loading U-Net semantic checkpoint: {UNET_SEMANTIC_CHECKPOINT}")
        state_dict = torch.load(UNET_SEMANTIC_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model

    if AUTO_TRAIN_UNET_IF_MISSING:
        model = train_unet_semantic_checkpoint()
        model.to(DEVICE)
        model.eval()
        return model

    raise FileNotFoundError(
        f"U-Net semantic checkpoint not found at {UNET_SEMANTIC_CHECKPOINT}"
    )


def unet_semantic_predict(model, image_np):
    image = Image.fromarray(image_np).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1)
        pred = pred.squeeze().cpu().numpy().astype(np.uint8)

    return pred


# =========================
# SEGFORMER SEMANTIC
# =========================

def load_segformer_semantic_model():
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

    processor = SegformerImageProcessor.from_pretrained(SEGFORMER_CHECKPOINT)
    model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_CHECKPOINT)

    model.to(DEVICE)
    model.eval()

    return processor, model


def segformer_semantic_predict(processor, model, image_np):
    image = Image.fromarray(image_np).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    pred = upsampled_logits.argmax(dim=1)
    pred = pred.squeeze().cpu().numpy().astype(np.uint8)

    # Safety:
    # If checkpoint has more than 21 classes, remove labels outside Pascal VOC range.
    # This does not make ADE20K labels equivalent to VOC labels.
    pred[pred >= NUM_CLASSES] = 0

    return pred


# =========================
# SAM CLASS-AGNOSTIC
# =========================

def load_sam_model():
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    if not os.path.exists(SAM_CHECKPOINT):
        raise FileNotFoundError(
            f"SAM checkpoint not found at {SAM_CHECKPOINT}. "
            "Place sam_vit_b_01ec64.pth in the project folder."
        )

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=100
    )

    return mask_generator


def sam_class_agnostic_predict(mask_generator, image_np):
    """
    SAM class-agnostic output:
    0 = background
    1 = mask/object region

    SAM does not output VOC semantic class IDs.
    """

    masks = mask_generator.generate(image_np)

    pred = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

    for mask_info in masks:
        mask = mask_info["segmentation"]
        pred[mask] = 1

    return pred


# =========================
# VISUALIZATION
# =========================

def create_visual_comparison(
    image_np,
    gt_binary,
    gt_semantic,
    predictions,
    save_path
):
    panels = ["Original", "GT Binary", "GT Semantic"] + list(predictions.keys())

    total_cols = len(panels)

    plt.figure(figsize=(4 * total_cols, 4))

    col = 1

    plt.subplot(1, total_cols, col)
    plt.imshow(image_np)
    plt.title("Original")
    plt.axis("off")
    col += 1

    plt.subplot(1, total_cols, col)
    plt.imshow(gt_binary, cmap="gray", vmin=0, vmax=1)
    plt.title("GT Binary")
    plt.axis("off")
    col += 1

    plt.subplot(1, total_cols, col)
    plt.imshow(gt_semantic, cmap="tab20", vmin=0, vmax=20)
    plt.title("GT Semantic")
    plt.axis("off")
    col += 1

    for method_name, pred_mask in predictions.items():
        plt.subplot(1, total_cols, col)

        if method_name in ["Watershed-Binary", "SAM-Class-Agnostic"]:
            plt.imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        else:
            plt.imshow(pred_mask, cmap="tab20", vmin=0, vmax=20)

        plt.title(method_name)
        plt.axis("off")
        col += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_bar_chart(summary_df, metric_name, save_path):
    plt.figure(figsize=(12, 5))

    labels = summary_df["method"] + " (" + summary_df["task"] + ")"

    plt.bar(labels, summary_df[metric_name])
    plt.xlabel("Method")
    plt.ylabel(metric_name)
    plt.title(f"Average {metric_name} by Method")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_all_metrics(summary_df, save_path):
    fig, ax = plt.subplots(figsize=(14, 6))

    labels = summary_df["method"] + " (" + summary_df["task"] + ")"
    x = np.arange(len(labels))
    width = 0.25

    ax.bar(x - width, summary_df["mean_mIoU"], width, label="mIoU")
    ax.bar(x, summary_df["mean_pixel_accuracy"], width, label="Pixel Accuracy")
    ax.bar(x + width, summary_df["mean_inference_time_sec"], width, label="Inference Time Sec")

    ax.set_xlabel("Method")
    ax.set_ylabel("Value")
    ax.set_title("Segmentation Evaluation Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# MAIN
# =========================

def main():
    set_seed(RANDOM_SEED)

    print(f"Using device: {DEVICE}")

    dataset = load_voc_dataset(image_set="val")

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected_indices = indices[:NUM_IMAGES]

    print(f"Selected {len(selected_indices)} images from Pascal VOC 2012 validation set.")

    unet_model = None
    segformer_processor = None
    segformer_model = None
    sam_mask_generator = None

    if USE_UNET_SEMANTIC:
        print("Loading U-Net Semantic...")
        unet_model = load_unet_semantic_model()

    if USE_SEGFORMER_SEMANTIC:
        print("Loading SegFormer Semantic...")
        segformer_processor, segformer_model = load_segformer_semantic_model()

    if USE_SAM_CLASS_AGNOSTIC:
        print("Loading SAM...")
        sam_mask_generator = load_sam_model()

    all_results = []

    visual_sample_count = 0
    max_visual_samples = 8

    for count, idx in enumerate(tqdm(selected_indices, desc="Evaluating")):
        image_pil, mask_pil = dataset[idx]

        image_np = preprocess_image_pil(image_pil, IMAGE_SIZE)
        gt_semantic = preprocess_mask_pil(mask_pil, IMAGE_SIZE)
        gt_binary = convert_voc_to_binary(gt_semantic)

        image_name = f"image_{count + 1:03d}"

        predictions_for_visual = {}

        # -------------------------
        # 1. Watershed Binary
        # -------------------------
        if USE_WATERSHED_BINARY:
            pred_binary, inference_time = measure_time(
                watershed_binary_predict,
                image_np
            )

            miou, pixel_acc = compute_binary_metrics(pred_binary, gt_binary)

            all_results.append({
                "image_id": image_name,
                "method": "Watershed-Binary",
                "task": "binary",
                "mIoU": miou,
                "pixel_accuracy": pixel_acc,
                "inference_time_sec": inference_time
            })

            predictions_for_visual["Watershed-Binary"] = pred_binary

        # -------------------------
        # 2. Watershed Semantic Oracle
        # -------------------------
        if USE_WATERSHED_SEMANTIC:
            pred_semantic, inference_time = measure_time(
                watershed_semantic_oracle_predict,
                image_np,
                gt_semantic
            )

            miou, pixel_acc = compute_semantic_metrics(
                pred_semantic,
                gt_semantic,
                NUM_CLASSES
            )

            all_results.append({
                "image_id": image_name,
                "method": "Watershed-Semantic-Oracle",
                "task": "semantic-oracle",
                "mIoU": miou,
                "pixel_accuracy": pixel_acc,
                "inference_time_sec": inference_time
            })

            predictions_for_visual["Watershed-Semantic"] = pred_semantic

        # -------------------------
        # 3. U-Net Semantic
        # -------------------------
        if USE_UNET_SEMANTIC:
            pred_semantic, inference_time = measure_time(
                unet_semantic_predict,
                unet_model,
                image_np
            )

            miou, pixel_acc = compute_semantic_metrics(
                pred_semantic,
                gt_semantic,
                NUM_CLASSES
            )

            all_results.append({
                "image_id": image_name,
                "method": "U-Net-Semantic",
                "task": "semantic",
                "mIoU": miou,
                "pixel_accuracy": pixel_acc,
                "inference_time_sec": inference_time
            })

            predictions_for_visual["U-Net-Semantic"] = pred_semantic

        # -------------------------
        # 4. SegFormer Semantic
        # -------------------------
        if USE_SEGFORMER_SEMANTIC:
            pred_semantic, inference_time = measure_time(
                segformer_semantic_predict,
                segformer_processor,
                segformer_model,
                image_np
            )

            miou, pixel_acc = compute_semantic_metrics(
                pred_semantic,
                gt_semantic,
                NUM_CLASSES
            )

            all_results.append({
                "image_id": image_name,
                "method": "SegFormer-Semantic",
                "task": "semantic",
                "mIoU": miou,
                "pixel_accuracy": pixel_acc,
                "inference_time_sec": inference_time
            })

            predictions_for_visual["SegFormer-Semantic"] = pred_semantic

        # -------------------------
        # 5. SAM Class-Agnostic
        # -------------------------
        if USE_SAM_CLASS_AGNOSTIC:
            pred_binary, inference_time = measure_time(
                sam_class_agnostic_predict,
                sam_mask_generator,
                image_np
            )

            miou, pixel_acc = compute_binary_metrics(
                pred_binary,
                gt_binary
            )

            all_results.append({
                "image_id": image_name,
                "method": "SAM-Class-Agnostic",
                "task": "class-agnostic-binary",
                "mIoU": miou,
                "pixel_accuracy": pixel_acc,
                "inference_time_sec": inference_time
            })

            predictions_for_visual["SAM-Class-Agnostic"] = pred_binary

        # -------------------------
        # Save visual examples
        # -------------------------
        if visual_sample_count < max_visual_samples:
            save_path = f"{RESULTS_DIR}/visualizations/{image_name}_comparison.png"

            create_visual_comparison(
                image_np=image_np,
                gt_binary=gt_binary,
                gt_semantic=gt_semantic,
                predictions=predictions_for_visual,
                save_path=save_path
            )

            visual_sample_count += 1

    # =========================
    # SAVE RESULTS
    # =========================

    results_df = pd.DataFrame(all_results)

    raw_results_path = f"{RESULTS_DIR}/quantitative_results.csv"
    results_df.to_csv(raw_results_path, index=False)

    summary_df = results_df.groupby(["method", "task"]).agg(
        mean_mIoU=("mIoU", "mean"),
        mean_pixel_accuracy=("pixel_accuracy", "mean"),
        mean_inference_time_sec=("inference_time_sec", "mean"),
        std_mIoU=("mIoU", "std"),
        std_pixel_accuracy=("pixel_accuracy", "std"),
        std_inference_time_sec=("inference_time_sec", "std")
    ).reset_index()

    summary_path = f"{RESULTS_DIR}/summary_results.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nEvaluation completed.")
    print("\nSummary Results:")
    print(summary_df)

    print(f"\nRaw results saved to: {raw_results_path}")
    print(f"Summary results saved to: {summary_path}")
    print(f"Visual outputs saved to: {RESULTS_DIR}/visualizations")

    # =========================
    # CHARTS
    # =========================

    plot_metric_bar_chart(
        summary_df,
        "mean_mIoU",
        f"{RESULTS_DIR}/charts/miou_bar_chart.png"
    )

    plot_metric_bar_chart(
        summary_df,
        "mean_pixel_accuracy",
        f"{RESULTS_DIR}/charts/pixel_accuracy_bar_chart.png"
    )

    plot_metric_bar_chart(
        summary_df,
        "mean_inference_time_sec",
        f"{RESULTS_DIR}/charts/inference_time_bar_chart.png"
    )

    plot_all_metrics(
        summary_df,
        f"{RESULTS_DIR}/charts/all_metrics_summary.png"
    )

    print(f"Charts saved to: {RESULTS_DIR}/charts")


if __name__ == "__main__":
    main()