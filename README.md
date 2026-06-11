# Segmentation Methods Evolution

This repository contains an end-semester research project for **Advanced Topics in AI**. The project compares the evolution of image segmentation methods from classical image-processing algorithms to CNN-based, transformer-based, and foundation-model approaches.

## Project Title

**The Evolution of Segmentation Methods: From Region-Based Algorithms to Transformers**

## Group Members

| Name |
|---|
| Zahad Ali Zafar 
| Zeeshan Jamil 
| Mehboob Ellahi 
| Bilal Zubair 

## Repository Link

https://github.com/Engr-Zeeshan-Jamil/segmentation-methods-evolution

## Overview

Image segmentation is a pixel-level computer vision task where each pixel is assigned to an object or background class. This project evaluates representative segmentation methods from different generations:

| Method | Type | Output |
|---|---|---|
| Watershed Binary | Classical image processing | Background / foreground |
| Watershed Semantic Oracle | Classical + oracle assignment | Pascal VOC semantic labels using majority ground-truth class |
| U-Net Semantic | CNN-based segmentation | 21 Pascal VOC classes |
| SegFormer Semantic | Transformer-based segmentation | Semantic segmentation output |
| SAM Class-Agnostic | Foundation model | Class-agnostic object masks |

The comparison is performed on a fixed subset of **100 images** from the Pascal VOC 2012 validation split.

## Evaluation Metrics

The following metrics are used:

| Metric | Meaning |
|---|---|
| mIoU | Mean Intersection-over-Union; measures overlap between prediction and ground truth |
| Pixel Accuracy | Percentage of correctly classified pixels |
| Inference Time | Average processing time per image |

## Final Results

| Method | Task | Mean mIoU | Mean Pixel Accuracy | Mean Inference Time |
|---|---|---:|---:|---:|
| SAM-Class-Agnostic | Class-agnostic binary | 0.2322 | 0.3774 | 25.1876 sec |
| SegFormer-Semantic | Semantic | 0.0543 | 0.2836 | 0.4958 sec |
| U-Net-Semantic | Semantic | 0.2231 | 0.7964 | 0.4544 sec |
| Watershed-Binary | Binary | 0.3622 | 0.5645 | 0.0194 sec |
| Watershed-Semantic-Oracle | Semantic oracle | 0.4472 | 0.7562 | 0.0257 sec |

## Key Observations

Watershed-Binary is the fastest method because it uses classical image-processing operations and does not require neural network inference.

Watershed-Semantic-Oracle achieves the highest mIoU, but this result should be interpreted carefully because it uses ground-truth majority-class assignment. It is not a deployable semantic segmentation model.

U-Net-Semantic achieves the highest pixel accuracy among the learned semantic models, but its mIoU remains moderate, showing that pixel accuracy can be inflated by dominant background regions.

SegFormer-Semantic produces visually meaningful regions in some qualitative examples, but the measured mIoU is low in this setup, likely due to checkpoint/class-label mismatch with Pascal VOC.

SAM-Class-Agnostic captures object-like regions but does not assign semantic class labels. It is also the slowest method because automatic mask generation is computationally expensive.

## Project Structure

```text
.
├── evaluate_segmentation_final.py
├── README.md
├── requirements.txt
├── results_final/
│   ├── charts/
│   ├── visualizations/
│   ├── quantitative_results.csv
│   └── summary_results.csv
└── .gitignore