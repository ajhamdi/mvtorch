## Installation


## Requirements

The MVTorch library is primarily written in PyTorch, with some components utilizing CUDA for enhanced performance. While it is possible to use MVTorch with a CPU, it is recommended to use it with a GPU in order to access all features

- install Pytorch3d from [here](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
```bash
conda create -y -n mvtorchenv python=3.7
conda activate mvtorchenv
conda install -c pytorch pytorch=1.8.0 torchvision cudatoolkit=10.2
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
``` 

- install other helper libraries 

```bash
conda install -y pandas
conda install -y -c conda-forge trimesh
pip install imageio einops scipy matplotlib ptflops tensorboard h5py pptk metric-learn
``` 

- install vision helper libraries [MMCV](https://mmcv.readthedocs.io/en/latest/) , mmsegmentation , timm , and detectron2 
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
pip install mmsegmentation # install the latest release
pip install git+https://github.com/rwightman/pytorch-image-models
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html
