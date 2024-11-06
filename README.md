# Nutrition Estimation with RGB-D Images

## Overview

This project is focused on predicting nutritional information (calories, protein, fat, carbohydrates) from food images using RGB-D (color + depth) data. Utilizing the Nutrition5K dataset and a deep learning model based on ResNet101, this project aims to provide accurate and efficient nutrition estimation for individual food items or dishes. The inclusion of depth data enhances portion size estimation and helps in handling overlapping food items.

## Dataset

The Nutrition5K dataset is used for training and evaluating the model. It provides paired RGB and depth images of various food items, labeled with comprehensive nutritional information. The dataset includes over 250 food categories and captures both color and spatial data for each food item.

To access and download the Nutrition5K dataset, please refer to [here](https://github.com/google-research-datasets/Nutrition5k).

## Model Architecture

The backbone model is based on ResNet101.

Train the model via `python train_RGBD_multi_fusion.py --model resnet101  --dataset nutrition_rgbd --rgbd  --direct_prediction`

## Requirements
torch

torchvision

tqdm

numpy

opencv-python


Contributors:\
Brady Huai (@BradyHuai)\
Wuchen Li (@SightVanish)\
Hejie Huang (@kikimasu1)\
Konard Kopko
