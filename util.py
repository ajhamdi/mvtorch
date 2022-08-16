import torch
import matplotlib.pyplot as plt
import imageio
import numpy as np

def batch_tensor(tensor, dim=1, squeeze=False):
    """
    A function to reshape PyTorch tensor `tensor` along some dimension `dim` to the batch dimension 0 such that the tensor can be processed in parallel. 
    If `sqeeze`=True, the dimension `dim` will be removed completely, otherwise it will be of size=1. Check `unbatch_tensor()` for the reverese function.
    """
    batch_size, dim_size = tensor.shape[0], tensor.shape[dim]
    returned_size = list(tensor.shape)
    returned_size[0] = batch_size * dim_size
    returned_size[dim] = 1
    if squeeze:
        return tensor.transpose(0, dim).reshape(returned_size).squeeze_(dim)
    else:
        return tensor.transpose(0, dim).reshape(returned_size)


def unbatch_tensor(tensor, batch_size, dim=1, unsqueeze=False):
    """
    A function to chunk pytorch tensor `tensor` along the batch dimension 0 and concatenate the chuncks on dimension `dim` to recover from `batch_tensor()` function.
    If `unsqueee`=True, it will add a dimension `dim` before the unbatching. 
    """
    fake_batch_size = tensor.shape[0]
    nb_chunks = int(fake_batch_size / batch_size)
    if unsqueeze:
        return torch.cat(torch.chunk(tensor.unsqueeze_(dim), nb_chunks, dim=0), dim=dim).contiguous()
    else:
        return torch.cat(torch.chunk(tensor, nb_chunks, dim=0), dim=dim).contiguous()

def check_valid_rotation_matrix(R, tol:float=1e-6):
    """
    Determine if R is a valid rotation matrix by checking it satisfies the
    following conditions:
    ``RR^T = I and det(R) = 1``
    Args:
        R: an (N, 3, 3) matrix
    Returns:
        None
    Emits a warning if R is an invalid rotation matrix.
    """
    N = R.shape[0]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    eye = eye.view(1, 3, 3).expand(N, -1, -1)
    orthogonal = torch.allclose(R.bmm(R.transpose(1, 2)), eye, atol=tol)
    det_R = torch.det(R)
    no_distortion = torch.allclose(det_R, torch.ones_like(det_R))
    return orthogonal and no_distortion

def torch_color(color_type, custom_color=(1.0,0,0), max_lightness=False, epsilon=0.00001):
    """
    A function to return a torch tesnor of size 3 that represents a color according to the 'color_type' string that can be [white, red, green, black, random, custom]. If max_lightness is true, color is normalized to be brightest.
    """
    if color_type == "white":
        color =  torch.tensor((1.0, 1.0, 1.0))
    if color_type == "red":
        color =  torch.tensor((1.0, 0.0, 0.0))
    if color_type == "green":
        color = torch.tensor((0.0, 1.0, 0.0))
    if color_type == "blue":
        color = torch.tensor((0.0, 0.0, 1.0))
    if color_type == "black":
        color = torch.tensor((0.0, 0.0, 0.0))
    elif color_type == "random":
        color = torch.rand(3)
    elif color =="custom":
        color = torch.tensor(custom_color)

    if max_lightness and color_type != "black":
        color = color / (torch.max(color) + epsilon)

    return color

def get_camera_wireframe(scale: float = 0.3):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * torch.tensor([-2, 1.5, 4])
    b = 0.5 * torch.tensor([2, 1.5, 4])
    c = 0.5 * torch.tensor([-2, -1.5, 4])
    d = 0.5 * torch.tensor([2, -1.5, 4])
    C = torch.zeros(3)
    F = torch.tensor([0, 0, 3])
    camera_points = [a, b, d, c, a, C, b, d, C, c, C, F]
    lines = torch.stack([x.float() for x in camera_points]) * scale
    return lines

def plot_cameras(ax, cameras, color: str = "blue", scale: float = 0.3):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """
    cam_wires_canonical = get_camera_wireframe(scale).cuda()[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
    plot_handles = []
    for wire in cam_wires_trans:
        # the Z and Y axes are flipped intentionally here!
        x_, z_, y_ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
    return plot_handles

def save_cameras(cameras, save_path, scale=0.22, dpi=200):
    import mpl_toolkits
    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.axes3d.Axes3D(fig)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_zlim(-1.8, 1.8)
    ax.scatter(xs=[0], ys=[0], zs=[0], linewidth=3, c="r")
    plot_cameras(ax, cameras, color="blue", scale=scale)
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


def save_grid(image_batch, save_path, **kwargs):
    """
    a hleper function for torchvision.util function `make_grid` to save a batch of images (B,H,W,C) as a grid on the `save_path` 
    """
    from torchvision.utils import make_grid
    im = make_grid(image_batch, **kwargs).detach().cpu().transpose(0, 2).transpose(0, 1).numpy()
    imageio.imsave(save_path, (255.0*im).astype(np.uint8))