import os
import time
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torchvision.datasets import VOCSegmentation
from torchvision import transforms

warnings.filterwarnings("ignore")


# =========================
# CONFIGURATION
# =========================

DATA_ROOT = "./data"
RESULTS_DIR = "./results"
NUM_IMAGES = 100
IMAGE_SIZE = 512
RANDOM_SEED = 42

USE_UNET = True
USE_SEGFORMER = True
USE_SAM = True

SAM_CHECKPOINT = "./sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/visualizations", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/charts", exist_ok=True)


# =========================
# UTILITY FUNCTIONS
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_voc_dataset():
    dataset = VOCSegmentation(
        root=DATA_ROOT,
        year="2012",
        image_set="val",
        download=False
    )
    return dataset


def preprocess_image_pil(image_pil, size=512):
    image_pil = image_pil.convert("RGB")
    original_size = image_pil.size

    image_resized = image_pil.resize((size, size), Image.BILINEAR)
    image_np = np.array(image_resized)

    return image_np, original_size


def preprocess_mask_pil(mask_pil, size=512):
    mask_np = np.array(mask_pil)

    mask_resized = Image.fromarray(mask_np).resize((size, size), Image.NEAREST)
    mask_resized = np.array(mask_resized)

    return mask_resized


def convert_voc_to_binary(mask):
    """
    Pascal VOC:
    0 = background
    1-20 = object classes
    255 = void / boundary

    This function converts semantic mask into:
    0 = background
    1 = foreground
    255 = ignore
    """
    binary = np.zeros_like(mask, dtype=np.uint8)
    binary[(mask > 0) & (mask != 255)] = 1
    binary[mask == 255] = 255
    return binary


def compute_binary_metrics(pred, gt):
    """
    Computes foreground/background mIoU and pixel accuracy.
    Ignores pixels with value 255.
    """

    valid_mask = gt != 255
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

        iou = intersection / union
        ious.append(iou)

    miou = np.mean(ious) if len(ious) > 0 else 0.0

    return float(miou), float(pixel_accuracy)


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
# WATERSHED MODEL
# =========================

def watershed_predict(image_np):
    """
    Classical segmentation using OpenCV watershed.
    Output:
    0 = background
    1 = foreground
    """

    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(
        dist_transform, 0.4 * dist_transform.max(), 255, 0
    )

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image_bgr, markers)

    pred = np.zeros(gray.shape, dtype=np.uint8)
    pred[markers > 1] = 1

    return pred


# =========================
# U-NET MODEL
# =========================

def load_unet_model():
    """
    Uses segmentation_models_pytorch U-Net.
    Note:
    This model is ImageNet-encoder pretrained, but decoder is not VOC-trained.
    For a proper academic comparison, replace this with a VOC-trained U-Net checkpoint.
    """

    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )

    model.to(DEVICE)
    model.eval()
    return model


def unet_predict(model, image_np):
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

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    return pred


# =========================
# SEGFORMER MODEL
# =========================

def load_segformer_model():
    """
    Loads SegFormer fine-tuned on ADE20K.
    Note:
    ADE20K labels do not perfectly match Pascal VOC labels.
    For strict VOC mIoU, use a SegFormer checkpoint fine-tuned on Pascal VOC.
    Here we convert output into binary foreground/background.
    """

    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

    checkpoint = "nvidia/segformer-b0-finetuned-ade-512-512"

    processor = SegformerImageProcessor.from_pretrained(checkpoint)
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint)

    model.to(DEVICE)
    model.eval()

    return processor, model


def segformer_predict(processor, model, image_np):
    image = Image.fromarray(image_np).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    semantic_pred = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

    # Convert semantic classes to foreground/background.
    # In ADE20K, class 0 can be background-like depending on label mapping.
    # Here, all non-zero predictions are treated as foreground.
    binary_pred = np.zeros_like(semantic_pred, dtype=np.uint8)
    binary_pred[semantic_pred > 0] = 1

    return binary_pred


# =========================
# SAM MODEL
# =========================

def load_sam_model():
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    if not os.path.exists(SAM_CHECKPOINT):
        raise FileNotFoundError(
            f"SAM checkpoint not found at {SAM_CHECKPOINT}. "
            "Download sam_vit_b_01ec64.pth and place it in the project folder."
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


def sam_predict(mask_generator, image_np):
    masks = mask_generator.generate(image_np)

    pred = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

    for mask_info in masks:
        mask = mask_info["segmentation"]
        pred[mask] = 1

    return pred


# =========================
# VISUALIZATION FUNCTIONS
# =========================

def create_visual_comparison(
    image_np,
    gt_binary,
    predictions,
    image_name,
    save_path
):
    methods = ["Original", "Ground Truth"] + list(predictions.keys())
    total_cols = len(methods)

    plt.figure(figsize=(4 * total_cols, 4))

    plt.subplot(1, total_cols, 1)
    plt.imshow(image_np)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, total_cols, 2)
    plt.imshow(gt_binary, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    col_index = 3
    for method_name, pred_mask in predictions.items():
        plt.subplot(1, total_cols, col_index)
        plt.imshow(pred_mask, cmap="gray")
        plt.title(method_name)
        plt.axis("off")
        col_index += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_bar_chart(results_df, metric_name, save_path):
    summary = results_df.groupby("method")[metric_name].mean().reset_index()

    plt.figure(figsize=(8, 5))
    plt.bar(summary["method"], summary[metric_name])
    plt.xlabel("Method")
    plt.ylabel(metric_name)
    plt.title(f"Average {metric_name} by Method")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_all_metrics(summary_df, save_path):
    metrics = ["mean_mIoU", "mean_pixel_accuracy", "mean_inference_time_sec"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(summary_df["method"]))
    width = 0.25

    ax.bar(x - width, summary_df["mean_mIoU"], width, label="mIoU")
    ax.bar(x, summary_df["mean_pixel_accuracy"], width, label="Pixel Accuracy")
    ax.bar(x + width, summary_df["mean_inference_time_sec"], width, label="Inference Time Sec")

    ax.set_xlabel("Method")
    ax.set_title("Segmentation Evaluation Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["method"], rotation=30)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# MAIN EVALUATION
# =========================

def main():
    set_seed(RANDOM_SEED)

    print(f"Using device: {DEVICE}")

    dataset = load_voc_dataset()

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected_indices = indices[:NUM_IMAGES]

    print(f"Selected {len(selected_indices)} images from Pascal VOC 2012 validation set.")

    unet_model = None
    segformer_processor = None
    segformer_model = None
    sam_mask_generator = None

    if USE_UNET:
        print("Loading U-Net...")
        unet_model = load_unet_model()

    if USE_SEGFORMER:
        print("Loading SegFormer...")
        segformer_processor, segformer_model = load_segformer_model()

    if USE_SAM:
        print("Loading SAM...")
        sam_mask_generator = load_sam_model()

    all_results = []

    visual_sample_count = 0
    max_visual_samples = 8

    for count, idx in enumerate(tqdm(selected_indices, desc="Evaluating")):
        image_pil, mask_pil = dataset[idx]

        image_np, _ = preprocess_image_pil(image_pil, IMAGE_SIZE)
        gt_mask = preprocess_mask_pil(mask_pil, IMAGE_SIZE)
        gt_binary = convert_voc_to_binary(gt_mask)

        image_name = f"image_{count + 1:03d}"

        predictions_for_visual = {}

        # -------------------------
        # Watershed
        # -------------------------
        pred, inference_time = measure_time(watershed_predict, image_np)
        miou, pixel_acc = compute_binary_metrics(pred, gt_binary)

        all_results.append({
            "image_id": image_name,
            "method": "Watershed",
            "mIoU": miou,
            "pixel_accuracy": pixel_acc,
            "inference_time_sec": inference_time
        })

        predictions_for_visual["Watershed"] = pred

        # -------------------------
        # U-Net
        # -------------------------
        if USE_UNET:
            pred, inference_time = measure_time(unet_predict, unet_model, image_np)
            miou, pixel_acc = compute_binary_metrics(pred, gt_binary)

            all_results.append({
                "image_id": image_name,
                "method": "U-Net",
                "mIoU": miou,
                "pixel_accuracy": pixel_acc,
                "inference_time_sec": inference_time
            })

            predictions_for_visual["U-Net"] = pred

        # -------------------------
        # SegFormer
        # -------------------------
        if USE_SEGFORMER:
            pred, inference_time = measure_time(
                segformer_predict,
                segformer_processor,
                segformer_model,
                image_np
            )
            miou, pixel_acc = compute_binary_metrics(pred, gt_binary)

            all_results.append({
                "image_id": image_name,
                "method": "SegFormer",
                "mIoU": miou,
                "pixel_accuracy": pixel_acc,
                "inference_time_sec": inference_time
            })

            predictions_for_visual["SegFormer"] = pred

        # -------------------------
        # SAM
        # -------------------------
        if USE_SAM:
            pred, inference_time = measure_time(sam_predict, sam_mask_generator, image_np)
            miou, pixel_acc = compute_binary_metrics(pred, gt_binary)

            all_results.append({
                "image_id": image_name,
                "method": "SAM",
                "mIoU": miou,
                "pixel_accuracy": pixel_acc,
                "inference_time_sec": inference_time
            })

            predictions_for_visual["SAM"] = pred

        # -------------------------
        # Save qualitative examples
        # -------------------------
        if visual_sample_count < max_visual_samples:
            save_path = f"{RESULTS_DIR}/visualizations/{image_name}_comparison.png"
            create_visual_comparison(
                image_np=image_np,
                gt_binary=gt_binary,
                predictions=predictions_for_visual,
                image_name=image_name,
                save_path=save_path
            )
            visual_sample_count += 1

    # =========================
    # SAVE RESULTS
    # =========================

    results_df = pd.DataFrame(all_results)

    raw_results_path = f"{RESULTS_DIR}/quantitative_results.csv"
    results_df.to_csv(raw_results_path, index=False)

    summary_df = results_df.groupby("method").agg(
        mean_mIoU=("mIoU", "mean"),
        mean_pixel_accuracy=("pixel_accuracy", "mean"),
        mean_inference_time_sec=("inference_time_sec", "mean"),
        std_mIoU=("mIoU", "std"),
        std_pixel_accuracy=("pixel_accuracy", "std"),
        std_inference_time_sec=("inference_time_sec", "std")
    ).reset_index()

    summary_path = f"{RESULTS_DIR}/summary_results.csv"
    summary_df.to_csv(summary_path, index=False)

    # Generate charts
    plot_metric_bar_chart(
        results_df,
        "mIoU",
        f"{RESULTS_DIR}/charts/miou_bar_chart.png"
    )

    plot_metric_bar_chart(
        results_df,
        "pixel_accuracy",
        f"{RESULTS_DIR}/charts/pixel_accuracy_bar_chart.png"
    )

    plot_metric_bar_chart(
        results_df,
        "inference_time_sec",
        f"{RESULTS_DIR}/charts/inference_time_bar_chart.png"
    )

    plot_all_metrics(
        summary_df,
        f"{RESULTS_DIR}/charts/all_metrics_summary.png"
    )

    print("\nEvaluation completed.")
    print("\nSummary Results:")
    print(summary_df)

    print(f"\nRaw results saved to: {raw_results_path}")
    print(f"Summary results saved to: {summary_path}")
    print(f"Visual outputs saved to: {RESULTS_DIR}/visualizations")
    print(f"Charts saved to: {RESULTS_DIR}/charts")


if __name__ == "__main__":
    main()