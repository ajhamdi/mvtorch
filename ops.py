from util import *

from pytorch3d.renderer import look_at_view_transform

import sys

def check_and_correct_rotation_matrix(R, T, nb_trials, azim, elev, dist):
    exhastion = 0
    while not check_valid_rotation_matrix(R):
            exhastion += 1
            R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(elev.T + 90.0 * torch.rand_like(elev.T, device=elev.device),
                                                                                                                                 dim=1, squeeze=True), azim=batch_tensor(azim.T + 180.0 * torch.rand_like(azim.T, device=elev.device), dim=1, squeeze=True))
            if not check_valid_rotation_matrix(R) and exhastion > nb_trials:
                sys.exit("Remedy did not work")
    return R , T