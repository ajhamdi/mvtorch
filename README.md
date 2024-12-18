<p align="center">
<img src="./docs/misc/logomvtorch.png" width="35%"/> 
<!-- <br> 
<h2 align="center" style="color:#e67300;font-size:40px;">MVTorch</2>  -->
</p>

# MVTorch [[paper](https://link.springer.com/article/10.1007/s11263-024-02283-5)]
A modular Pytroch library for multi-view research on 3D understanding and 3D generation. It is published as part of the [MVTN IJCV Journal paper](https://link.springer.com/article/10.1007/s11263-024-02283-5)
## Introduction

MVTorch provides efficient, reusable components for 3D Computer Vision and Graphics research based on mult-view representation with [PyTorch](https://pytorch.org) and [Pytorch3D](https://github.com/facebookresearch/pytorch3d).

### Key Features include:

- Render differentiable multi-view images from meshes and point clouds with 3D-2D correspondances.
- Data loaders for 3D data and multi-view images (posed or unposed )
- Visualizations of 3D mesh,point cloud, multi-view images. 
- Modular training of multi-view networks for different 3D tasks 
- I/O 3D data and multi-view images. 


### Benifits :

- Are implemented using PyTorch tensors and on top of Pytorch3D 
- Can handle minibatches of hetereogenous data
- Can be differentiated for input gradients.
- Can utilize GPUs for acceleration



## Installation

<!-- ```bibtex
pip install mvtorch
```
 -->
For detailed instructions refer to [INSTALL.md](INSTALL.md).

## Test

- After installing `mvtorch`, download common 3D datasets ([ModelNet40](https://mega.nz/file/mm5FhJ7I#jGECWn-QSCLH9LLoxhZzSWnf9LCtCavV12toj9SJKPM), [ScanObjectNN](https://mega.nz/file/ampg2QyT#Exo22r-8jzgCa2MOqoqipd39HVqYKG5iykJ5bovjsuI), [ShapeNet Parts](https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip), [nerf_synthetic](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi)) and unzip inside `data` directory.

```bash
cd data/
wget https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip --no-check-certificate # download ShapeNet Parts
# download the other datasets from the browser
```
- Run any example from `examples` directory 
```bibtex
cd examples/ && python classification.py 
```
## Tutorials

Get started with MVTorch by trying one of the following tutorials.


|<img src="./docs/misc/cls.png" width="310" height="310"/> | <img src="./docs/misc/seg.png" width="310" height="310"/>|
|:-----------------------------------------------------------------------------------------------------------:|:--------------------------------------------------:|
| [Training MVCNN in 10 lines of code for 3D Classification](https://github.com/ajhamdi/mvtorch/blob/main/docs/tutorials/classification.ipynb)| [Training 3D Part Segmentation with Multi-View DeepLabV3](https://github.com/ajhamdi/mvtorch/blob/main/docs/tutorials/segmentation.ipynb) |


|<img src="https://user-images.githubusercontent.com/7057863/78473103-9353b300-7770-11ea-98ed-6ba2d877b62c.gif" width="378" height="378"/> | <img src="https://github.com/threedle/text2mesh/blob/main/images/vases.gif" width="378"/>|
|:-----------------------------------------------------------------------------------------------------------:|:--------------------------------------------------:|
| [               Fit A Simple Neural Radiance Field                       ](https://github.com/ajhamdi/mvtorch/blob/main/docs/tutorials/nerf.ipynb)| [              Create Textured Meshes from Text              ](https://github.com/ajhamdi/mvtorch/blob/main/docs/tutorials/text2mesh.ipynb) |



### Key Classes
- [**MVRenderer**](https://github.com/ajhamdi/mvtorch/tree/fc83d72c1f43e977b61db91984eb6731bdcaaed6/mvtorch/mvrenderer.py#L25) ( renders multi-view images of both point clouds and meshes )
- [**MVNetwork**](https://github.com/ajhamdi/mvtorch/tree/fc83d72c1f43e977b61db91984eb6731bdcaaed6/mvtorch/networks.py#L6) ( allow to take any 2D network as input and outputs its multi-view features)
- [**Visualizer**](https://github.com/ajhamdi/mvtorch/tree/fc83d72c1f43e977b61db91984eb6731bdcaaed6/mvtorch/visualizer.py#L4)  ( handles multi-view and 3D visualization both for server saves and interactive visualization)
- [**data I/O**](https://github.com/ajhamdi/mvtorch/blob/main/mvtorch/data.py) ( load any dataset: modelnet, shapenet, scanobjectnn, shapenet parts, s3dis, nerf, as well as saving Multi-view datasets.)
- [**ViewSelector**](https://github.com/ajhamdi/mvtorch/tree/fc83d72c1f43e977b61db91984eb6731bdcaaed6/mvtorch/view_selector.py#L300) ( multi-view selector to select M viewpoints to render: random, circular ,spherical, [mvtn](https://github.com/ajhamdi/MVTN)  etc ... )
- [**MVAggregate**](https://github.com/ajhamdi/mvtorch/blob/fc83d72c1f43e977b61db91984eb6731bdcaaed6/mvtorch/mvaggregate.py#L70) ( a super model that accepts any 2D network as input and outputs the global multi-view features of input multi-view images: MeanPool, MaxPool) 
- [**MVLifting**](https://github.com/ajhamdi/mvtorch/blob/fc83d72c1f43e977b61db91984eb6731bdcaaed6/mvtorch/mvaggregate.py#L196) ( aggregates dense features from multi-view pixel features to 3D features  , eg. LabelPool, MeanPool, [Voint](https://arxiv.org/abs/2111.15363) aggregation and lifting ) 
- other useful utility functions and operations.

## Development

We welcome new contributions to MVTorch by following this procedure for pull requests: 

- For code modifications, create an issue with tag `request` and wait for 10 days for the issue to be resolved.

- If issue not resolved in 10 days, fork the repo and create a pull request on a new branch. Please make sure the main [examples](https://github.com/ajhamdi/mvtorch/tree/main/examples) can run after your adjustments on the core library.

- For additional examples, just create a pull request without creating an issue. 

- If you can contribue regularly on the library, please contact [Abdullah](abdullah.hamdi@kaust.edu.sa) to be added to the contruters list.



## Citation

If you find mvtorch useful in your research, please cite the library paper:

```bibtex
@article{Hamdi2024,
  author    = {Abdullah Hamdi and Faisal AlZahrani and Silvio Giancola and Bernard Ghanem},
  title     = {MVTN: Learning Multi-view Transformations for 3D Understanding},
  journal   = {International Journal of Computer Vision},
  year      = {2024},
  doi       = {10.1007/s11263-024-02283-5},
  issn      = {1573-1405}
}
``` 

## News

**[July 23 2022]:**   MVTorch repo created

**[December 26 2022]:**   MVTorch made public


## Projects
Projects that MVTorch benifited from in devlopment: [MVTN](https://arxiv.org/abs/2011.13244), [Voint Cloud](https://arxiv.org/abs/2111.15363), [Text2Mesh](https://github.com/threedle/text2mesh) and [NeRF](https://github.com/yenchenlin/nerf-pytorch)

## Documentation

A detailed documentation of the library should be coming soon... 

### Overview Video
Coming soon ...

## License

MVTorch is released under the [BSD License](LICENSE).
