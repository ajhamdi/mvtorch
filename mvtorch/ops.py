from .utils import *

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

def knn(x, k):
    """
    Given point features x [B, C, N, 1], and number of neighbors k (int)
    Return the idx for the k neighbors of each point. 
    So, the shape of idx: [B, N, k]
    """
    with torch.no_grad():
        x = x.squeeze(-1)
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        inner = -xx - inner - xx.transpose(2, 1)

        idx = inner.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def post_process_segmentation(point_set, predictions_3d, iterations=1, K_neighbors=1):
    """
    a function to fill empty points in point cloud `point_set` with the labels of their nearest neghbors in `predictions_3d` in an iterative fashion 
    """
    for iter in range(iterations):
        emptry_points = predictions_3d == 0
        nbr_indx = knn(point_set.transpose(
            1, 2)[..., None], iter*K_neighbors + 2)
        nbr_labels = batched_index_select_(
            predictions_3d[..., None].transpose(1, 2)[..., None], nbr_indx)
        # only look at the closest neighbor to fetch its labels
        nbr_labels = torch.mode(nbr_labels[:, 0, :, 1::], dim=-1)[0]
        predictions_3d[emptry_points] = nbr_labels[emptry_points]
    return predictions_3d