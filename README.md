# Key Information Extraction with LayoutLM

This repository contains source files for a project that focuses on key information extraction from forms/documents using LayoutLM models. 

## Overview

Extracting key information from various forms and documents is a common challenge. This project aims to automate this process using various models such as LayoutLM.

## Features

- Efficient key information extraction from specific forms and document types.

## Getting Started
For using the LayoutLM either in `layoutxlm.ipynb` or `layoutxlm_experiments.py` u have tu setup detectron environment in conda (https://github.com/facebookresearch/detectron2). For YOLOv8 u have to install ultralytics and pylabel packeges.

## Using
### Yolo
U firstly have to download the labeled data from Label Studio in YOLO format and set dataset_dir in `yolov8.ipynb` to path of ur dataset. Then u can run the code.

### LayoutLM
Running experiments for layoutLM is in `layoutxlm_experiments.py`, you have to prepare the dataset which contains images folder, CSV anotations from label studio, ALTO format of OCR transcription for each image and u have to have Wandb account which u set in source code or u can just run `layoutxlm.ipynb` for demo example.
