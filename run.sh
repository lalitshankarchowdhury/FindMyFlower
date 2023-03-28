#!/bin/bash
python train.py flowers/ --gpu
python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --gpu