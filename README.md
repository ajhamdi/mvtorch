# MVTorch
a Pytorch library for multi-view 3D understanding.
# Introduction

MVTorch provides efficient, reusable components for 3D Computer Vision and Graphics research based on mult-view representation with [PyTorch](https://pytorch.org) and [Pytorch3D](https://github.com/facebookresearch/pytorch3d).

## Key Features include:

- Render differentiable  multi-view images from meshes and point clouds
- I/O 3D data and multi-view images. 
- Data loaders for 3D data 
- Visualizations of 3D mesh,point cloud, multi-view images. 
- Modular training of multi-view networks for different 3D tasks 


## Benifits :

- Are implemented using PyTorch tensors and on top of 
- Can handle minibatches of hetereogenous data
- Can be differentiated
- Can utilize GPUs for acceleration

Projects that use MVTorch:  [MVTN](https://arxiv.org/abs/2011.13244) and [Voint Cloud](https://arxiv.org/abs/2111.15363).

## Installation

```bibtex
pip install mvtorch
```
For detailed instructions refer to [INSTALL.md](INSTALL.md).

## License

MVTorch is released under the [BSD License](LICENSE).

## Tutorials

Get started with MVTorch by trying one of the following examples.

| [Training MVCNN in 10 lines of code for ModelNet40 classification](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/deform_source_mesh_to_target_mesh.ipynb)| [Training segmentation on ShapeNetParts  with MVCNN ](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/bundle_adjustment.ipynb) |

| <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/render_textured_mesh.gif" width="310"/> | <img src="https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/.github/camera_position_teapot.gif" width="310" height="310"/>
|:------------------------------------------------------------:|:--------------------------------------------------:|




## Documentation

Learn more about the API by reading ...

### Overview Video

## Development

We welcome new contributions to MVTorch by following this procedure for pull requests: 

- ...

- ...


## Citation

If you find mvtorch useful in your research, please cite the extended paper:

```bibtex

```

```bibtex
@InProceedings{Hamdi_2021_ICCV,
    author    = {Hamdi, Abdullah and Giancola, Silvio and Ghanem, Bernard},
    title     = {MVTN: Multi-View Transformation Network for 3D Shape Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {1-11}
}
```

## News

**[July 23 2022]:**   MVTorch repo created
