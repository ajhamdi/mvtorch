# to import files from parent dir
import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvtorch.data import load_nerf_data
from mvtorch.models.nerf import *
import numpy as np
import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')

datadir = './data/nerf_synthetic/chair' # Input data directory
testskip = 8 # Load 1/testskip images from test/val sets
white_bkgd = True # Render synthetic data on a white background
render_test = False # Render the test set instead of render_poses path
N_rand = 32*32 # Batch size (number of random rays per gradient step)
no_batching = True # Only take random rays from 1 image at a time
i_print = 100 # Frequency of console printout
lrate = 5e-4 # Learning rate
lrate_decay = 500 # Exponential learning rate decay (in 1000 steps)
multires = 10 # log2 of max freq for positional encoding (3D location)
i_embed = 0 # Set 0 for default positional encoding, -1 for none
use_viewdirs = True # Use full 5D input instead of 3D
multires_views = 4 # log2 of max freq for positional encoding (2D direction)
N_importance = 128 # Number of additional fine samples per ray
netdepth = 8 # Layers in network
netwidth = 256 # Channels per layer
netdepth_fine = 8 # Layers in fine network
netwidth_fine = 256 # Channels per layer in fine network
chunk = 1024*32 # Number of rays processed in parallel, decrease if running out of memory
netchunk = 1024*64 # Number of pts sent through network in parallel, decrease if running out of memory
N_samples = 64 # Number of coarse samples per ray
perturb = 1. # Set to 0. for no jitter, 1. for jitter
raw_noise_std = 0. # Std dev of noise added to regularize sigma_a output, 1e0 recommended
no_ndc = True # do not use normalized device coordinates (set for non-forward facing scenes)
lindisp = False # Sampling linearly in disparity rather than depth
precrop_iters = 500 # Number of steps to train on central crops
precrop_frac = .5 # Fraction of img taken for central crops
i_video = 50000 # Frequency of render_poses video saving
i_testset = 50000 # Frequency of testset saving
basedir = './results/' # Where to store ckpts and logs
expname = 'test' # Experiment name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set torch device

# Load data
K = None
images, poses, render_poses, hwf, i_split = load_nerf_data(datadir, testskip)
print('Loaded nerf data', images.shape, render_poses.shape, hwf, datadir)
i_train, i_val, i_test = i_split

near = 2.
far = 6.

if white_bkgd:
    images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
else:
    images = images[...,:3]

# Cast intrinsics to right types
H, W, focal = hwf
H, W = int(H), int(W)
hwf = [H, W, focal]

if K is None:
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

if render_test:
    render_poses = np.array(poses[i_test])

# Create nerf model
render_kwargs_train, render_kwargs_test, grad_vars, optimizer = create_nerf(
    multires=multires,
    i_embed=i_embed,
    use_viewdirs=use_viewdirs,
    multires_views=multires_views,
    N_importance=N_importance,
    netdepth=netdepth,
    netwidth=netwidth,
    device=device,
    netdepth_fine=netdepth_fine,
    netwidth_fine=netwidth_fine,
    lrate=lrate,
    netchunk=netchunk,
    white_bkgd=white_bkgd,
    N_samples=N_samples,
    perturb=perturb,
    raw_noise_std=raw_noise_std,
    no_ndc=no_ndc,
    lindisp=lindisp,
)
global_step = 0

bds_dict = {
    'near' : near,
    'far' : far,
}
render_kwargs_train.update(bds_dict)
render_kwargs_test.update(bds_dict)

# Move testing data to GPU
render_poses = torch.Tensor(render_poses).to(device)

# Prepare raybatch tensor if batching random rays
N_rand = N_rand
use_batching = not no_batching
if use_batching:
    # For random ray batching
    print('get rays')
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
    print('done, concats')
    rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
    rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    print('shuffle rays')
    np.random.shuffle(rays_rgb)

    print('done')
    i_batch = 0

# Move training data to GPU
if use_batching:
    images = torch.Tensor(images).to(device)
poses = torch.Tensor(poses).to(device)
if use_batching:
    rays_rgb = torch.Tensor(rays_rgb).to(device)


N_iters = 200000 + 1
print('Begin')
print('TRAIN views are', i_train)
print('TEST views are', i_test)
print('VAL views are', i_val)

start = global_step + 1
for i in range(start, N_iters):
    # Sample random ray batch
    if use_batching:
        # Random over all images
        batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]

        i_batch += N_rand
        if i_batch >= rays_rgb.shape[0]:
            print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            i_batch = 0

    else:
        # Random from one image
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3,:4]

        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

            if i < precrop_iters:
                dH = int(H//2 * precrop_frac)
                dW = int(W//2 * precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {precrop_iters}")                
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

    #####  Core optimization loop  #####
    rgb, disp, acc, extras = render(H, W, K, chunk=chunk, rays=batch_rays,
                                            verbose=i < 10, retraw=True,
                                            **render_kwargs_train)

    optimizer.zero_grad()
    img_loss = img2mse(rgb, target_s)
    trans = extras['raw'][...,-1]
    loss = img_loss
    psnr = mse2psnr(img_loss)

    if 'rgb0' in extras:
        img_loss0 = img2mse(extras['rgb0'], target_s)
        loss = loss + img_loss0
        psnr0 = mse2psnr(img_loss0)

    loss.backward()
    optimizer.step()

    # NOTE: IMPORTANT!
    ###   update learning rate   ###
    decay_rate = 0.1
    decay_steps = lrate_decay * 1000
    new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate
    ################################


    if i % i_video == 0 and i > 0:
        if not os.path.exists(os.path.join(basedir, expname)):
            os.makedirs(os.path.join(basedir, expname))
        # Turn on testing mode
        with torch.no_grad():
            rgbs, disps = render_path(render_poses, hwf, K, chunk, render_kwargs_test)
        print('Done, saving', rgbs.shape, disps.shape)
        moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
        imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
    
    if i % i_testset == 0 and i > 0:
        if not os.path.exists(os.path.join(basedir, expname)):
            os.makedirs(os.path.join(basedir, expname))
        testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', poses[i_test].shape)
        with torch.no_grad():
            render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
        print('Saved test set')

    if i % i_print == 0:
        print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

    global_step += 1