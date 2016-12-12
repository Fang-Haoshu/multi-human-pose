# RMPE: Regional Multi-person Pose Estimation(SPPE+SSTN part)

This repository includes Torch code for training our SPPE+SSTN model presented in our [arXiv paper](https://arxiv.org/abs/1612.00137v1).


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

To install STN module, do
  ```Shell
  git clone https://github.com/Fang-Haoshu/stnbhwd.git
  cd stnbhwd && luarocks make stnbhwd-scm-1.rockspec
  ```

After all the installation is done, get the code. We will call the directory that you cloned this repo into `$SPPE_ROOT`
  ```Shell
  git clone https://github.com/Fang-Haoshu/multi-human-pose.git
  cd multi-human-pose
  ```


### Preparation
#### For evaluation only
1. Download pre-trained SPPE+SSTN model([Google drive](https://drive.google.com/open?id=0BxORzoJl8N9PSThkWUVNS0xSc3c)|[Baidu cloud](http://pan.baidu.com/s/1eS5edRc)). By default, we assume the models are stored in `$SPPE_ROOT/predict/model/`.

#### For training
1. Download [stacked hourglass networks](http://www-personal.umich.edu/~alnewell/pose/umich-stacked-hourglass.zip). By default, we assume the model is stored in `$SPPE_ROOT/train/`.

2. Download [MPII images](http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz) and [COCO14 training set](http://msvocds.blob.core.windows.net/coco2014/train2014.zip). By default, we assume the images are stored in `/data/MPII_COCO14/images/`.


### Evaluate
You will need to generate bounding box first. Here we have already generated the bounding box in `$SPPE_ROOT/predict/annot/mpii-test0.09/`. To generate it yourself, please follow the guidlines in the main [repo]().

  ```Shell
  cd $SPPE_ROOT/predict
  # make a soft link to the images then test
  ln -s /data/MPII_COCO14/images/ data/images
  # get the predicted results
  th main.lua predict-test
  ```
You can also use the following command to visualize the single person pose results.
```Shell
  qlua main.lua demo
  ```
### Train
We finetune our model based on the pre-trained stacked-hourglass model.
  ```Shell
  cd $SPPE_ROOT/train/src
  ln -s /data/MPII_COCO14/images ../data/mpii/images
  # First finetune the model using DPG
  th main.lua -expID finetune -addDPG
  th main.lua -expID finetune -addDPG -continue -LR 0.5e-4 -nEpochs 10
  # Then add parallel SPPE and SSTN on the finetuned model
  # It should reach a final mAP of 80.*
  th main.lua -expID final_model -loadModel '../exp/mpii/finetune/final_model.t7' -LR 0.5e-4 -addParallelSPPE -addSTN -addDPG
  th main.lua -expID final_model -continue -Npochs 8 -LR 0.1e-4 -addParallelSPPE -addSTN -addDPG
  ```


### Acknowledgements

Thanks to [Wei Liu](https://github.com/weiliu89/caffe/tree/ssd), [Alejandro Newell](https://github.com/anewell/pose-hg-train), [Pfister, T.](https://github.com/tpfister/caffe-heatmap), [Kaichun Mo](https://github.com/daerduoCarey/SpatialTransformerLayer), [Maxime Oquab](https://github.com/qassemoquab/stnbhwd) for contributing their codes. 
Thanks to the authors of Caffe and Torch7!
