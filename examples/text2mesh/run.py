# to import files from parent dir
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import clip
from tqdm import tqdm
import torch
import numpy as np
import copy
import torchvision
import os
from torchvision import transforms
from mvtorch.utils import torch_color
from mvtorch.mvrenderer import MVRenderer
from utils import Mesh, MeshNormalizer, NeuralStyleField, device
from pytorch3d.structures import Meshes

obj_path = '../examples/text2mesh/meshes/candle.obj'
# prompt = 'an image of a candle made of colorful crochet'
prompt = 'a candle made of colorful wood'
clipmodel = 'ViT-B/32'
dir = '../examples/text2mesh/outputs/'
n_augs = 1
background = torch_color('white').to(device)
cropforward = False
normmincrop = 0.1
normmaxcrop = 0.1
mincrop = 1
maxcrop = 1
cropsteps = 0
n_iter = 1500
input_normals = False
only_z = False
sigma = 5.0
depth = 4
width = 256
colordepth = 2
normdepth = 2
normratio = 0.1
clamp = 'tanh'
normclamp = 'tanh'
pe = True
exclude = 0
learning_rate = 0.0005
decay = 0
lr_decay = 0.9
decay_step = 100
lr_plateau = False
symmetry = False
standardize = False
n_views = 5
show = False
clipavg = 'view'
splitnormloss = False
splitcolorloss = False
geoloss = True
n_normaugs = 4
decayfreq = None
cropdecay = 1.0
frontview_center = [1.96349, 0.6283] # azim, elev
frontview_std = 4
render_dist = 2

def report_process(dir, i, loss, loss_check, losses, rendered_images):
    print('iter: {} loss: {}'.format(i, loss.item()))
    torchvision.utils.save_image(rendered_images, os.path.join(dir, 'iter_{}.jpg'.format(i)))
    if lr_plateau and loss_check is not None:
        new_loss_check = np.mean(losses[-100:])
        # If avg loss increased or plateaued then reduce LR
        if new_loss_check >= loss_check:
            for g in torch.optim.param_groups:
                g['lr'] *= 0.5
        loss_check = new_loss_check

    elif lr_plateau and loss_check is None and len(losses) >= 100:
        loss_check = np.mean(losses[-100:])

def export_final_results(dir, losses, mesh, mlp, network_input, vertices):
    with torch.no_grad():
        pred_rgb, pred_normal = mlp(network_input)
        pred_rgb = pred_rgb.detach().cpu()
        pred_normal = pred_normal.detach().cpu()

        torch.save(pred_rgb, os.path.join(dir, f"colors_final.pt"))
        torch.save(pred_normal, os.path.join(dir, f"normals_final.pt"))

        base_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5)
        final_color = torch.clamp(pred_rgb + base_color, 0, 1)

        mesh.vertices = vertices.detach().cpu() + mesh.vertex_normals.detach().cpu() * pred_normal

        objbase, extension = os.path.splitext(os.path.basename(obj_path))
        mesh.export(os.path.join(dir, f"{objbase}_final.obj"), color=final_color)

        # Save final losses
        torch.save(torch.tensor(losses), os.path.join(dir, "losses.pt"))


def update_mesh(mlp, network_input, prior_color, sampled_mesh, vertices):
    pred_rgb, pred_normal = mlp(network_input)
    sampled_mesh.face_attributes = prior_color + pred_rgb
    sampled_mesh.vertices = vertices + sampled_mesh.vertex_normals * pred_normal
    MeshNormalizer(sampled_mesh)()


# Load CLIP model 
clip_model, preprocess = clip.load(clipmodel, device)
res = clip_model.visual.input_resolution
    
objbase, extension = os.path.splitext(os.path.basename(obj_path))

# render = Renderer(dim=(res, res))
render = MVRenderer(n_views, image_size=res, pc_rendering=False, background_color=background, object_color='custom', return_mapping=False)
mesh = Mesh(obj_path)
MeshNormalizer(mesh)()

prior_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5, device=device)

losses = []

clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
# CLIP Transform
clip_transform = transforms.Compose([
    transforms.Resize((res, res)),
    clip_normalizer
])

# Augmentation settings
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(res, scale=(1, 1)),
    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
    clip_normalizer
])

# Augmentations for normal network
if cropforward :
    curcrop = normmincrop
else:
    curcrop = normmaxcrop
normaugment_transform = transforms.Compose([
    transforms.RandomResizedCrop(res, scale=(curcrop, curcrop)),
    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
    clip_normalizer
])
cropiter = 0
cropupdate = 0
if normmincrop < normmaxcrop and cropsteps > 0:
    cropiter = round(n_iter / (cropsteps + 1))
    cropupdate = (maxcrop - mincrop) / cropiter

    if not cropforward:
        cropupdate *= -1

# Displacement-only augmentations
displaugment_transform = transforms.Compose([
    transforms.RandomResizedCrop(res, scale=(normmincrop, normmincrop)),
    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
    clip_normalizer
])

normweight = 1.0

# MLP Settings
input_dim = 6 if input_normals else 3
if only_z:
    input_dim = 1
mlp = NeuralStyleField(sigma, depth, width, 'gaussian', colordepth, normdepth,
                            normratio, clamp, normclamp, niter=n_iter,
                            progressive_encoding=pe, input_dim=input_dim, exclude=exclude).to(device)
mlp.reset_weights()

optim = torch.optim.Adam(mlp.parameters(), learning_rate, weight_decay=decay)
activate_scheduler = lr_decay < 1 and decay_step > 0 and not lr_plateau
if activate_scheduler:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=decay_step, gamma=lr_decay)


prompt = ' '.join(prompt)
prompt_token = clip.tokenize([prompt]).to(device)
encoded_text = clip_model.encode_text(prompt_token)

# Save prompt
with open(os.path.join(dir, prompt), "w") as f:
    f.write("")

# Same with normprompt
norm_encoded = encoded_text


loss_check = None
vertices = copy.deepcopy(mesh.vertices)
network_input = copy.deepcopy(vertices)
if symmetry:
    network_input[:,2] = torch.abs(network_input[:,2])

if standardize:
    # Each channel into z-score
    network_input = (network_input - torch.mean(network_input, dim=0))/torch.std(network_input, dim=0)

for i in tqdm(range(n_iter)):
    optim.zero_grad()

    sampled_mesh = mesh

    update_mesh(mlp, network_input, prior_color, sampled_mesh, vertices)

    elev = torch.cat((torch.tensor([frontview_center[1]]), torch.randn(n_views - 1) * np.pi / frontview_std + frontview_center[1]))
    azim = torch.cat((torch.tensor([frontview_center[0]]), torch.randn(n_views - 1) * 2 * np.pi / frontview_std + frontview_center[0]))
    dist = torch.ones((n_views), dtype=torch.float) * render_dist
    color = sampled_mesh.face_attributes
    azim, elev, dist, color = azim.unsqueeze(0), elev.unsqueeze(0), dist.unsqueeze(0), color.unsqueeze(0)
    rendered_images = render(Meshes(verts=[sampled_mesh.vertices], faces=[sampled_mesh.faces], verts_normals=[sampled_mesh.vertex_normals]), None, azim, elev, dist, color)[0].squeeze(0)

    if n_augs == 0:
        clip_image = clip_transform(rendered_images)
        encoded_renders = clip_model.encode_image(clip_image)
        loss = torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))

    # Check augmentation steps
    if cropsteps != 0 and cropupdate != 0 and i != 0 and i % cropsteps == 0:
        curcrop += cropupdate
        normaugment_transform = transforms.Compose([
            transforms.RandomResizedCrop(res, scale=(curcrop, curcrop)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer
        ])

    if n_augs > 0:
        loss = 0.0
        for _ in range(n_augs):
            augmented_image = augment_transform(rendered_images)
            encoded_renders = clip_model.encode_image(augmented_image)
            if clipavg == "view":
                if encoded_text.shape[0] > 1:
                    loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                    torch.mean(encoded_text, dim=0), dim=0)
                else:
                    loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                    encoded_text)
            else:
                loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
    if splitnormloss:
        for param in mlp.mlp_normal.parameters():
            param.requires_grad = False
    loss.backward(retain_graph=True)


    if n_normaugs > 0:
        normloss = 0.0
        for _ in range(n_normaugs):
            augmented_image = normaugment_transform(rendered_images)
            encoded_renders = clip_model.encode_image(augmented_image)
            if clipavg == "view":
                if norm_encoded.shape[0] > 1:
                    normloss -= normweight * torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                        torch.mean(norm_encoded, dim=0),
                                                                        dim=0)
                else:
                    normloss -= normweight * torch.cosine_similarity(
                        torch.mean(encoded_renders, dim=0, keepdim=True),
                        norm_encoded)
            else:
                normloss -= normweight * torch.mean(
                    torch.cosine_similarity(encoded_renders, norm_encoded))
        if splitnormloss:
            for param in mlp.mlp_normal.parameters():
                param.requires_grad = True
        if splitcolorloss:
            for param in mlp.mlp_rgb.parameters():
                param.requires_grad = False
        
        normloss.backward(retain_graph=True)

    # Also run separate loss on the uncolored displacements
    if geoloss:
        default_color = torch.zeros(len(mesh.vertices), 3).to(device)
        default_color[:, :] = torch.tensor([0.5, 0.5, 0.5]).to(device)
        sampled_mesh.face_attributes = default_color

        elev = torch.cat((torch.tensor([frontview_center[1]]), torch.randn(n_views - 1) * np.pi / frontview_std + frontview_center[1]))
        azim = torch.cat((torch.tensor([frontview_center[0]]), torch.randn(n_views - 1) * 2 * np.pi / frontview_std + frontview_center[0]))
        dist = torch.ones((n_views), dtype=torch.float) * render_dist
        color = sampled_mesh.face_attributes
        azim, elev, dist, color = azim.unsqueeze(0), elev.unsqueeze(0), dist.unsqueeze(0), color.unsqueeze(0)
        geo_renders = render(Meshes(verts=[sampled_mesh.vertices], faces=[sampled_mesh.faces], verts_normals=[sampled_mesh.vertex_normals]), None, azim, elev, dist, color)[0].squeeze(0)

        if n_normaugs > 0:
            normloss = 0.0
            ### avgview != aug
            for _ in range(n_normaugs):
                augmented_image = displaugment_transform(geo_renders)
                encoded_renders = clip_model.encode_image(augmented_image)
                if norm_encoded.shape[0] > 1:
                    normloss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                        torch.mean(norm_encoded, dim=0), dim=0)
                else:
                    normloss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                        norm_encoded)

            normloss.backward(retain_graph=True)
    
    optim.step()

    for param in mlp.mlp_normal.parameters():
        param.requires_grad = True
    for param in mlp.mlp_rgb.parameters():
        param.requires_grad = True

    if activate_scheduler:
        lr_scheduler.step()

    with torch.no_grad():
        losses.append(loss.item())

    # Adjust normweight if set
    if decayfreq is not None:
        if i % decayfreq == 0:
            normweight *= cropdecay

    if i % 100 == 0:
        report_process(dir, i, loss, loss_check, losses, rendered_images)

export_final_results(dir, losses, mesh, mlp, network_input, vertices)
