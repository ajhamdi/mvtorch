import torch
import matplotlib.pyplot as plt
import imageio
import numpy as np
import pickle
import os

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
    elif color_type == "red":
        color =  torch.tensor((1.0, 0.0, 0.0))
    elif color_type == "green":
        color = torch.tensor((0.0, 1.0, 0.0))
    elif color_type == "blue":
        color = torch.tensor((0.0, 0.0, 1.0))
    elif color_type == "black":
        color = torch.tensor((0.0, 0.0, 0.0))
    elif color_type == "random":
        color = torch.rand(3)
    else:
        if torch.is_tensor(color_type):
            color = color_type
        else:
            color = torch.tensor(color_type)

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

def torch_center_and_normalize(points,p="inf"):
    """
    a helper pytorch function that normalize and center 3D points clouds 
    """
    N = points.shape[0]
    center = points.mean(0)
    if p != "fro" and p!= "no":
        scale = torch.max(torch.norm(points - center, p=float(p),dim=1))
    elif p=="fro" :
        scale = torch.norm(points - center, p=p )
    elif p=="no":
        scale = 1.0
    points = points - center.expand(N, 3)
    points = points * (1.0 / float(scale))
    return points

def sort_jointly(list_of_arrays, dim=0):
    """
    sort all the arrays in `list_of_arrays` according to the sorting of the array `array list_of_arrays`[dim]
    """
    def swapPositions(mylsit, pos1, pos2):
        mylsit[pos1], mylsit[pos2] = mylsit[pos2], mylsit[pos1]
        return mylsit
    sorted_tuples = sorted(zip(*swapPositions(list_of_arrays, 0, dim)))
    combined_sorted = list(zip(*sorted_tuples))
    return [list(ii) for ii in swapPositions(combined_sorted, 0, dim)]

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def load_text(file_name):
    """
    a helper funcion to load text as lines and return a list of lines without `\n`
    """
    if not os.path.isfile(file_name):
        raise NameError("The file {} does not exisit".format(file_name))
    f = open(file_name, "r")
    lines = f.readlines()
    lines = [line.replace("\n", "") for line in lines]
    f.close()
    return lines

def torch_deg2rad(degs):
    return degs * np.pi/180.0

def torch_direction_vector(azim, elev, from_degrees=True):
    """
    a torch util fuinction to convert batch elevation and zimuth angles ( in degrees or radians) to a R^3 direction unit vector
    """
    bs = azim.shape[0]

    if from_degrees:
        azim, elev = torch_deg2rad(azim), torch_deg2rad(elev)
    dir_vector = torch.zeros(bs, 3)
    dir_vector[:, 0] = torch.sin(azim) * torch.cos(elev)
    dir_vector[:, 1] = torch.sin(elev)
    dir_vector[:, 2] = torch.cos(azim) * torch.cos(elev)
    return dir_vector

def class_freq_to_weight(class_freqs, alpha=1.0):
    """
    a function to convert  a dictionary of labels frequency  to dictionary of loss weights per class label that are averaged to 1. This is helpful in designing a weighted loss 
    """
    total = 0
    result_weights = {}
    cls_nbrs = len(class_freqs)
    for k, v in class_freqs.items():
        total += v
    avg = total/float(cls_nbrs)
    for k, v in class_freqs.items():
        result_weights[k] = alpha * avg/v + (1-alpha)
    return result_weights

def labels2freq(label_map):
    """
    a torch util funtion to return dict with frquencies of labels in the given examples in the case of dense labels.
    """
    lbl, inv_map, freq = torch.unique(
        label_map, return_counts=True, sorted=True, return_inverse=True)

    return {k: v for k, v in zip(lbl.detach().cpu().numpy().tolist(), freq.detach().cpu().numpy().tolist())}, inv_map

# from the nerf paper 
def positional_encoding(
    tensor, num_encoding_functions=0, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

def unit_spherical_grid(nb_points, return_radian=False, return_vertices=False):
    """
    a function that samples a grid of sinze `nb_points` around a sphere of radius `r` . it returns azimth and elevation angels arouns the sphere. if `return_vertices` is true .. it returns the 3d points as well 
    """
    r = 1.0
    vertices = []
    azim = []
    elev = []
    alpha = 4.0*np.pi*r*r/nb_points
    d = np.sqrt(alpha)
    m_nu = int(np.round(np.pi/d))
    d_nu = np.pi/m_nu
    d_phi = alpha/d_nu
    count = 0
    for m in range(0, m_nu):
        nu = np.pi*(m+0.5)/m_nu
        m_phi = int(np.round(2*np.pi*np.sin(nu)/d_phi))
        for n in range(0, m_phi):
            phi = 2*np.pi*n/m_phi
            xp = r*np.sin(nu)*np.cos(phi)
            yp = r*np.sin(nu)*np.sin(phi)
            zp = r*np.cos(nu)
            vertices.append([xp, yp, zp])
            azim.append(phi)
            elev.append(nu-np.pi*0.5)
            count = count + 1
    if not return_radian:
        azim = np.rad2deg(azim)
        elev = np.rad2deg(elev)
    if return_vertices:
        return azim[:nb_points], elev[:nb_points], np.array(vertices[:nb_points])
    else:
        return azim[:nb_points], elev[:nb_points]

def torch_direction_vector(azim, elev, from_degrees=True):
    """
    a torch util fuinction to convert batch elevation and zimuth angles ( in degrees or radians) to a R^3 direction unit vector
    """
    bs = azim.shape[0]

    if from_degrees:
        azim, elev = torch_deg2rad(azim), torch_deg2rad(elev)
    dir_vector = torch.zeros(bs, 3)
    dir_vector[:, 0] = torch.sin(azim) * torch.cos(elev)
    dir_vector[:, 1] = torch.sin(elev)
    dir_vector[:, 2] = torch.cos(azim) * torch.cos(elev)
    return dir_vector

def unbatch_tensor(tensor, batch_size, dim=1, unsqueeze=False):
    """
    a function to chunk pytorch tensor `tensor` along the batch dimension 0 and cincatenate the chuncks on dimension `dim` to recover from `batch_tensor()` function.
    if `unsqueee`=True , it will add a dimension `dim` before the unbatching 
    """
    fake_batch_size = tensor.shape[0]
    nb_chunks = int(fake_batch_size / batch_size)
    if unsqueeze:
        return torch.cat(torch.chunk(tensor.unsqueeze_(dim), nb_chunks, dim=0), dim=dim).contiguous()
    else:
        return torch.cat(torch.chunk(tensor, nb_chunks, dim=0), dim=dim).contiguous()