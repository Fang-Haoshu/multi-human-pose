# RMPE: Regional Multi-person Pose Estimation(SPPE+SSTN part)

This repository includes Torch code for training our SPPE+SSTN model presented in:

A pretrained model is available on the [project site](http://www-personal.umich.edu/~alnewell/pose). Include the model in the main directory of this repository to run the demo code.

**Check out the training and experimentation code now available at: [https://github.com/anewell/pose-hg-train](https://github.com/anewell/pose-hg-train)**

In addition, if you download the full [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de) and replace this repository's `images` directory you can generate full predictions on the validation and test sets.



### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Demo](#demo)
3. [Train/Eval](#traineval)
4. [Models](#models)

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

After all the installation is done, get the code. We will call the directory that you cloned Caffe into `$SPPE_ROOT`
  ```Shell
  git clone https://github.com/weiliu89/caffe.git
  cd caffe
  ```


### Preparation
#### For evaluate only
1. Download pre-trained [SPPE+SSTN model](). By default, we assume the models are stored in `$SPPE_ROOT/predict/model/`.

#### For training
1. Download [stacked hourglass networks](http://www-personal.umich.edu/~alnewell/pose/umich-stacked-hourglass.zip). By default, we assume the model is stored in `$SPPE_ROOT/train/`.

2. Download [MPII images](http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz) and [COCO14 training set](http://msvocds.blob.core.windows.net/coco2014/train2014.zip). By default, we assume the images are stored in `/data/MPII_COCO14/images/`.

3. Download [MPII_COCO14 Annotations](). By default, we assume the XMLs are stored in the `/data/MPII_COCO14/Annotations/`.

### Demo
Our experiments use both Caffe and Torch7. But we implement the whole framework in Caffe so you can run the demo easily.
1. Run the ipython notebook. It will show you how our whole framework works.
  ```Shell
  cd $CAFFE_ROOT
  # make a soft link to the images
  ln -s /data/MPII_COCO14/images/ data/MPII/images
  jupyter notebook examples/rmpe/Regional Multi-person Pose Estimation.ipynb
  ```

2. Run the python program for more results.
  ```Shell
  python examples/rmpe/demo.py
  ```

### Train/Eval
1. Train human detector. 
We use the data in MPII and COCO14 to train our human detector. We have already create the train/val list in `CAFFE_ROOT/data/MPII_COCO14` and release our script in `CAFFE_ROOT/examples/rmpe`, so basically what you need to do will be something like
  ```Shell
  # First create the LMDB file.
  cd $CAFFE_ROOT
  # You can modify the parameters in create_data.sh if needed.
  # It will create lmdb files for trainval and test with encoded original image:
  #   - /data/MPII_COCO14/lmdb/MPII_COCO14_trainval_lmdb
  #   - /data/MPII_COCO14/lmdb/MPII_COCO14_test_lmdb
  # and make soft links at examples/MPII_COCO14/
  ./data/MPII_COCO14/create_data.sh
  # It will create model definition files and save snapshot models in:
  #   - $CAFFE_ROOT/models/VGG_SSD/MPII_COCO14/SSD_500x500/
  # and job file, log file, and the python script in:
  #   - $CAFFE_ROOT/jobs/VGG_SSD/MPII_COCO14/SSD_500x500/
  # and save temporary evaluation results in:
  #   - $HOME/data/MPII_COCO14/results/SSD_500x500/
  # It should reach 85.* mAP at 60k iterations.
  python examples/rmpe/ssd_pascal_MPII_COCO14VGG.py
  ```

2. Train SPPE+SSTN.
This part of our model is implemented in Torch7. Please refer to [this repo]() for more details.
We will call the directory that you cloned the repo into `$SPPE_ROOT`.
Note that I am currently working on an implementation in Caffe. The script may come out soon.


3. Evaluate the model. You can modify line 45 in `demo.py` to evaluate our framework on whole test set. But the results may be slightly different from our work. To reproduce our results reported in our paper:
  ```Shell
  # First get the result of human detector
  cd $CAFFE_ROOT
  jupyter notebook examples/rmpe/human_detection.ipynb
  # Then move the results to $SPPE_ROOT/predict/annot/
  mv examples/rmpe/mpii-test0.09 $SPPE_ROOT/predict/annot/
  # Next, do single person human estimation
  cd $SPPE_ROOT/predict
  th main.lua predict-test
  #Finally, do pose NMS and write results to .mat
  python batch_nms.py

  ```

## Acknowledgements ##

Thanks to [Wei Liu](https://github.com/weiliu89/caffe/tree/ssd), [Alejandro Newell](https://github.com/anewell/pose-hg-train), [Pfister, T.](https://github.com/tpfister/caffe-heatmap), [Kaichun Mo](https://github.com/daerduoCarey/SpatialTransformerLayer), [Maxime Oquab](https://github.com/qassemoquab/stnbhwd) for contributing their codes. 
Thanks to the authors of Caffe and Torch7!