# DISCONTINUATION OF PROJECT #  
This project will no longer be maintained by Intel.  
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.  
Intel no longer accepts patches to this project.  
 If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project.  
  
# Stable View Synthesis

Code repository for "Stable View Synthesis".


## Setup

Install the following Python packages in your Python environment

```
- numpy (1.19.1)
- scikit-image (0.15.0)
- pillow (7.2.0)
- torch
- torchvision (0.7.0)
- torch-scatter (1.6)
- torch-sparse (1.6)
- torch-geometric (1.6)
- torch-sparse (1.6)
- open3d (0.11)
- python-opencv
- matplotlib (3.2.x)
- pandas (1.0.x)
```

To compile the Python extensions you will also need `Eigen` and `cmake`.

Clone the repository and initialize the submodule

```bash
git clone https://github.com/intel-isl/StableViewSynthesis.git
cd StableViewSynthesis
git submodule update --init --recursive
```

Finally, build the Python extensions

```
cd ext/preprocess
cmake -DCMAKE_BUILD_TYPE=Release .
make 

cd ../mytorch
python setup.py build_ext --inplace
```

Tested with Ubuntu 18.04 and macOS Catalina.


## Run Stable View Synthesis

Make sure you adapted the paths in `config.py` to point to the downloaded data!

```bash
cd experiments
```

Then run the evaluation via 

```bash
python exp.py --net resunet3.16_penone.dirs.avg.seq+9+1+unet+5+2+16.single+mlpdir+mean+3+64+16 --cmd eval --iter last --eval-dsets tat-subseq
```

This will run the pretrained network on the four Tanks and Temples sequences.

To train the network from scratch you can run

```bash
python exp.py --net resunet3.16_penone.dirs.avg.seq+9+1+unet+5+2+16.single+mlpdir+mean+3+64+16 --cmd retrain
```


## Data

See [FreeViewSynthesis](https://github.com/intel-isl/FreeViewSynthesis).

## Citation

Please cite our [paper](https://arxiv.org/abs/2011.07233) if you find this work useful.

```bib
@inproceedings{Riegler2021SVS,
  title={Stable View Synthesis},
  author={Riegler, Gernot and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

## Video

[![Stable View Synthesis Video](https://img.youtube.com/vi/gqgXIY09htI/0.jpg)](https://www.youtube.com/watch?v=gqgXIY09htI&feature=youtu.be)
