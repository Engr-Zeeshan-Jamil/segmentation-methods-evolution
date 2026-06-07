# Evolution of Image Segmentation Methods

This project compares the evolution of image segmentation methods from classical region-based algorithms to deep learning and foundation models.

## Methods Compared

- Watershed
- U-Net
- SegFormer
- Segment Anything Model (SAM)

## Dataset

The experiment uses a 100-image subset of the Pascal VOC 2012 validation dataset.

Dataset is not included in this repository. Download Pascal VOC 2012 separately and place it in:

```text
data/VOCdevkit/VOC2012
SAM Checkpoint
## SAM Checkpoint

## The SAM checkpoint is not included because of file size.

Download:

sam_vit_b_01ec64.pth

and place it in the project root folder.

## Installation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
Run Evaluation
python evaluate_segmentation.py
Outputs

The script generates:

quantitative_results.csv
summary_results.csv
metric charts
qualitative segmentation visualizations

## These outputs are saved in:

results/
Evaluation Metrics
Mean Intersection over Union
Pixel Accuracy
Inference Time per Image
Note

Watershed and SAM are not native semantic segmentation models. Therefore, outputs are converted into foreground/background masks for binary comparison against Pascal VOC ground truth.