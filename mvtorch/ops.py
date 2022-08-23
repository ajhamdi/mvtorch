from .util import *

from pytorch3d.renderer import look_at_view_transform

import sys

from einops import rearrange

def mvctosvc(x): 
    return rearrange(x, 'b m c h w -> (b m) c h w ')
def svctomvc(x,nb_views=1): 
    return rearrange(x, '(b m) c h w -> b m c h w',m=nb_views)

def check_and_correct_rotation_matrix(R, T, nb_trials, azim, elev, dist):
    exhastion = 0
    while not check_valid_rotation_matrix(R):
            exhastion += 1
            R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(elev.T + 90.0 * torch.rand_like(elev.T, device=elev.device),
                                                                                                                                 dim=1, squeeze=True), azim=batch_tensor(azim.T + 180.0 * torch.rand_like(azim.T, device=elev.device), dim=1, squeeze=True))
            if not check_valid_rotation_matrix(R) and exhastion > nb_trials:
                sys.exit("Remedy did not work")
    return R , T

def batched_index_select_(x, idx):
    """
    This can be used for neighbors features fetching
    Given a pointcloud x, return its k neighbors features indicated by a tensor idx.
    :param x: torch.Size([batch_size, num_dims, num_vertices, 1])
    :param index: torch.Size([batch_size, num_vertices, k])
    :return: torch.Size([batch_size, num_dims, num_vertices, k])
    """

    batch_size, num_dims, num_vertices = x.shape[:3]
    _, all_combo, k = idx.shape
    idx_base = torch.arange(
        0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_vertices, -1)[idx, :]
    feature = feature.view(batch_size, all_combo, k,
                           num_dims).permute(0, 3, 1, 2)
    return feature


def batched_index_select_parts(x, idx):
    """
    This can be used for neighbors features fetching
    Given a pointcloud x, return its k neighbors features indicated by a tensor idx.
    :param x: torch.Size([batch_size, num_vertices , 1])
    :param index: torch.Size([batch_size, num_views, points_per_pixel,H,W])
    :return: torch.Size([batch_size, _vertices, k])
    """

    batch_size, num_view, num_nbrs, H, W = idx.shape[:5]
    _, num_dims, num_vertices = x.shape

    idx = rearrange(idx, 'b m p h w -> b (m h w) p')
    x = x[..., None]
    feature = batched_index_select_(x, idx)
    feature = rearrange(feature, 'b d (m h w) p -> b m d p h w',
                        m=num_view, h=H, w=W, d=num_dims)
    return feature