## Installation


## Requirements

The MVTorch library is primarily written in PyTorch, with some components utilizing CUDA for enhanced performance. While it is possible to use MVTorch with a CPU, it is recommended to use it with a GPU in order to access all features

1. install `Pytorch3d` (depending on your system from [here](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md))
```bash
conda create -n mvtorchenv python=3.9
conda activate mvtorchenv
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
``` 

2. install `mvtorch` 

```bash
pip install git+https://github.com/ajhamdi/mvtorch
``` 