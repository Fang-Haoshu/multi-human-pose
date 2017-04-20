# RMPE: Regional Multi-person Pose Estimation(SPPE+SSTN part)

This repository includes Torch code for training our SPPE+SSTN model presented in our paper


### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Evaluate](#evaluate)
3. [Train](#train)
4. [Acknowledgements](#acknowledgements)

### Installation
To run this code, the following must be installed:

- [Torch7](https://github.com/torch/torch7)
- hdf5 (and the [torch-hdf5](https://github.com/deepmind/torch-hdf5/) package)
- cudnn
- qlua (for displaying results)
- STN module(use this [stable repo](https://github.com/Fang-Haoshu/stnbhwd))


After all the installation is done, get the code. We will call the directory that you cloned this repo into `$SPPE_ROOT`
  ```Shell
  git clone https://github.com/Fang-Haoshu/multi-human-pose.git
  cd multi-human-pose
  ```


### Preparation
#### For evaluation only
1. Download pre-trained SPPE+SSTN model([Baidu cloud](https://pan.baidu.com/s/1i4LJn97)). By default, we assume the models are stored in `$SPPE_ROOT/predict/`.

#### For training
1. Download [base networks](https://pan.baidu.com/s/1qYBBNYW) and [4-stack parallel SPPE](https://pan.baidu.com/s/1eRWEFUq). By default, we assume the models are stored in `$SPPE_ROOT/train/src`.

2. Download [MPII images](http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz). By default, we assume the images are stored in `/data/MPII/images/`.


### Evaluate
You will need to generate bounding box first. Here we have already generated the bounding boxes in `$SPPE_ROOT/predict/annot/mpii-test0.09/`. To generate it yourself, please follow the guidlines in the main [repo](https://github.com/Fang-Haoshu/RMPE#traineval).

  ```Shell
  cd $SPPE_ROOT/predict
  # make a soft link to the images
  ln -s /data/MPII/images/ data/images
  # get the predicted results
  th main.lua predict-test
  ```
You can also use the following command to visualize the single person pose estimation results.
```Shell
  qlua main.lua demo
  ```
### Train
We finetune our model based on the pre-trained stacked-hourglass model.
  ```Shell
  cd $SPPE_ROOT/train/src
  ln -s /data/MPII/images ../data/mpii-box/images
  # First finetune the model using PGPG
  th main.lua -expID finetune -usePGPG
  th main.lua -expID finetune -usePGPG -continue -LR 0.5e-4 -nEpochs 10
  # Then add parallel SPPE and SSTN on the finetuned model
  # It should reach a final mAP of 80.*
  th main.lua -expID final_model -loadModel '../exp/mpii-box/finetune/final_model.t7' -LR 0.5e-4 -addParallelSPPE -addSSTN -usePGPG
  th main.lua -expID final_model -continue -nEpochs 10 -LR 0.1e-4 -addParallelSPPE -addSSTN -usePGPG
  ```


### Acknowledgements

Thanks to [Wei Liu](https://github.com/weiliu89/caffe/tree/ssd), [Alejandro Newell](https://github.com/anewell/pose-hg-train), [Pfister, T.](https://github.com/tpfister/caffe-heatmap), [Kaichun Mo](https://github.com/daerduoCarey/SpatialTransformerLayer), [Maxime Oquab](https://github.com/qassemoquab/stnbhwd) for contributing their codes. 
Thanks to the authors of Caffe and Torch7!
