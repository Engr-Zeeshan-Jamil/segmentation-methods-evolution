# Evolution of Image Segmentation Methods

This project evaluates the evolution of image segmentation methods from classical region-based algorithms to modern deep learning and foundation models.

## Project Overview

The project compares four representative segmentation approaches:

- Watershed for classical region-based segmentation
- U-Net for CNN-based segmentation
- SegFormer for transformer-based segmentation
- Segment Anything Model (SAM) for foundation-model-based segmentation

The evaluation is performed on a 100-image subset of the Pascal VOC 2012 validation dataset.

## Objective

The objective of this project is to compare segmentation methods across different eras and analyze their performance in terms of:

- Mean Intersection over Union
- Pixel Accuracy
- Inference Time per Image
- Visual mask quality

## Dataset

This project uses the Pascal VOC 2012 dataset.

The dataset is not uploaded to this repository because of its size.

Expected dataset path:

```text
data/VOCdevkit/VOC2012
```

Expected folder structure:

```text
Project/
├── evaluate_segmentation.py
├── requirements.txt
├── data/
│   └── VOCdevkit/
│       └── VOC2012/
│           ├── JPEGImages/
│           ├── SegmentationClass/
│           ├── ImageSets/
│           ├── Annotations/
│           └── SegmentationObject/
```

## SAM Checkpoint

The SAM checkpoint is not included in this repository because it is a large file.

Download the SAM ViT-B checkpoint:

```text
sam_vit_b_01ec64.pth
```

Place it in the project root folder:

```text
Project/
├── evaluate_segmentation.py
├── requirements.txt
├── sam_vit_b_01ec64.pth
└── data/
```

The script expects the checkpoint here:

```python
SAM_CHECKPOINT = "./sam_vit_b_01ec64.pth"
```

## Installation

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Install SAM:

```powershell
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Run the Project

Run the evaluation script:

```powershell
python evaluate_segmentation.py
```

## Output

The script generates results inside the `results/` folder:

```text
results/
├── quantitative_results.csv
├── summary_results.csv
├── charts/
└── visualizations/
```

The output includes:

- Quantitative metric results
- Summary comparison table
- mIoU chart
- Pixel accuracy chart
- Inference time chart
- Qualitative mask comparison images

## Evaluation Metrics

The models are evaluated using:

- Mean Intersection over Union
- Pixel Accuracy
- Average inference time per image

## Important Note

Watershed and SAM are not native semantic segmentation models.

Watershed produces region-based segments without semantic class labels, while SAM produces class-agnostic object masks. Therefore, all outputs are converted into foreground/background segmentation masks for binary comparison against Pascal VOC ground-truth masks.

The reported mIoU and pixel accuracy reflect binary object segmentation performance rather than full 21-class Pascal VOC semantic segmentation performance.
