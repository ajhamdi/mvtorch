import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import os
import copy
import PIL
import numpy as np
from pytorch3d import io
from pytorch3d.structures import Meshes

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)

class ProgressiveEncoding(nn.Module):
    def __init__(self, mapping_size, T, d=3, apply=True):
        super(ProgressiveEncoding, self).__init__()
        self._t = 0
        self.n = mapping_size
        self.T = T
        self.d = d
        self._tau = 2 * self.n / self.T
        self.indices = torch.tensor([i for i in range(self.n)], device=device)
        self.apply = apply
    def forward(self, x):
        alpha = ((self._t - self._tau * self.indices) / self._tau).clamp(0, 1).repeat(
            2)  # no need to reduce d or to check cases
        if not self.apply:
            alpha = torch.ones_like(alpha, device=device)  ## this layer means pure ffn without progress.
        alpha = torch.cat([torch.ones(self.d, device=device), alpha], dim=0)
        self._t += 1
        return x * alpha


class NeuralStyleField(nn.Module):
    # Same base then split into two separate modules 
    def __init__(self, sigma, depth, width, encoding, colordepth=2, normdepth=2, normratio=0.1, clamp=None,
                 normclamp=None,niter=6000, input_dim=3, progressive_encoding=True, exclude=0):
        super(NeuralStyleField, self).__init__()
        self.pe = ProgressiveEncoding(mapping_size=width, T=niter, d=input_dim)
        self.clamp = clamp
        self.normclamp = normclamp
        self.normratio = normratio
        layers = []
        if encoding == 'gaussian':
            layers.append(FourierFeatureTransform(input_dim, width, sigma, exclude))
            if progressive_encoding:
                layers.append(self.pe)
            layers.append(nn.Linear(width * 2 + input_dim, width))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Linear(input_dim, width))
            layers.append(nn.ReLU())
        for i in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        self.base = nn.ModuleList(layers)

        # Branches 
        color_layers = []
        for _ in range(colordepth):
            color_layers.append(nn.Linear(width, width))
            color_layers.append(nn.ReLU())
        color_layers.append(nn.Linear(width, 3))
        self.mlp_rgb = nn.ModuleList(color_layers)

        normal_layers = []
        for _ in range(normdepth):
            normal_layers.append(nn.Linear(width, width))
            normal_layers.append(nn.ReLU())
        normal_layers.append(nn.Linear(width, 1))
        self.mlp_normal = nn.ModuleList(normal_layers)

        # print(self.base)
        # print(self.mlp_rgb)
        # print(self.mlp_normal)

    def reset_weights(self):
        self.mlp_rgb[-1].weight.data.zero_()
        self.mlp_rgb[-1].bias.data.zero_()
        self.mlp_normal[-1].weight.data.zero_()
        self.mlp_normal[-1].bias.data.zero_()

    def forward(self, x):
        for layer in self.base:
            x = layer(x)
        colors = self.mlp_rgb[0](x)
        for layer in self.mlp_rgb[1:]:
            colors = layer(colors)
        displ = self.mlp_normal[0](x)
        for layer in self.mlp_normal[1:]:
            displ = layer(displ)

        if self.clamp == "tanh":
            colors = torch.tanh(colors) / 2
        elif self.clamp == "clamp":
            colors = torch.clamp(colors, 0, 1)
        if self.normclamp == "tanh":
            displ = torch.tanh(displ) * self.normratio
        elif self.normclamp == "clamp":
            displ = torch.clamp(displ, -self.normratio, self.normratio)

        return colors, displ

def save_model(model, loss, iter, optim, output_dir):
    save_dict = {
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss
    }

    path = os.path.join(output_dir, 'checkpoint.pth.tar')

    torch.save(save_dict, path)

class Normalizer:
    @classmethod
    def get_bounding_box_normalizer(cls, x):
        shift = torch.mean(x, dim=0)
        scale = torch.max(torch.norm(x-shift, p=1, dim=1))
        return Normalizer(scale=scale, shift=shift)

    @classmethod
    def get_bounding_sphere_normalizer(cls, x):
        shift = torch.mean(x, dim=0)
        scale = torch.max(torch.norm(x-shift, p=2, dim=1))
        return Normalizer(scale=scale, shift=shift)

    def __init__(self, scale, shift):
        self._scale = scale
        self._shift = shift

    def __call__(self, x):
        return (x-self._shift) / self._scale

    def get_de_normalizer(self):
        inv_scale = 1 / self._scale
        inv_shift = -self._shift / self._scale
        return Normalizer(scale=inv_scale, shift=inv_shift)

class MeshNormalizer:
    def __init__(self, mesh):
        self._mesh = mesh  # original copy of the mesh
        self.normalizer = Normalizer.get_bounding_sphere_normalizer(self._mesh.vertices)

    def __call__(self):
        self._mesh.vertices = self.normalizer(self._mesh.vertices)
        return self._mesh

class FourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10, exclude=0):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self.exclude = exclude
        B = torch.randn((num_input_channels, mapping_size)) * scale
        B_sort = sorted(B, key=lambda x: torch.norm(x, p=2))
        self._B = torch.stack(B_sort)  # for sape

    def forward(self, x):
        # assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels = x.shape

        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        # x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        res = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        # x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        # x = x.permute(0, 3, 1, 2)

        res = 2 * np.pi * res
        return torch.cat([x, torch.sin(res), torch.cos(res)], dim=1)

class Mesh():
    def __init__(self,obj_path,color=torch.tensor([0.0,0.0,1.0])):
        if ".obj" in obj_path:
            # mesh = kal.io.obj.import_mesh(obj_path, with_normals=True)
            verts, faces, aux = io.load_obj(obj_path)
            mesh = Meshes(verts=[verts], faces=[faces.verts_idx], verts_normals=aux.normals.unsqueeze(0))
        elif ".off" in obj_path:
            # mesh = kal.io.off.import_mesh(obj_path)
            mesh = io.IO.load_mesh(path=obj_path)
        else:
            raise ValueError(f"{obj_path} extension not implemented in mesh reader.")
        self.meshes_object = mesh
        # self.vertices = mesh.vertices.to(device)
        self.vertices = mesh.verts_packed().to(device)
        # self.faces = mesh.faces.to(device)
        self.faces = mesh.faces_packed().to(device)
        self.vertex_normals = None
        self.face_normals = None
        self.texture_map = None
        self.face_uvs = None
        if ".obj" in obj_path:
            # if mesh.vertex_normals is not None:
            if mesh.verts_normals_packed() is not None:
                # self.vertex_normals = mesh.vertex_normals.to(device).float()
                self.vertex_normals = mesh.verts_normals_packed().to(device).float()

                # Normalize
                self.vertex_normals = torch.nn.functional.normalize(self.vertex_normals)

            # if mesh.face_normals is not None:
            if mesh.faces_normals_packed() is not None:
                # self.face_normals = mesh.face_normals.to(device).float()
                self.face_normals = mesh.faces_normals_packed().to(device).float()

                # Normalize
                self.face_normals = torch.nn.functional.normalize(self.face_normals)

        self.set_mesh_color(color)

    def standardize_mesh(self,inplace=False):
        mesh = self if inplace else copy.deepcopy(self)
        return standardize_mesh(mesh)

    def normalize_mesh(self,inplace=False):

        mesh = self if inplace else copy.deepcopy(self)
        return normalize_mesh(mesh)

    def update_vertex(self,verts,inplace=False):

        mesh = self if inplace else copy.deepcopy(self)
        mesh.vertices = verts
        return mesh

    def set_mesh_color(self,color):
        self.texture_map = get_texture_map_from_color(self,color)
        self.face_attributes = get_face_attributes_from_color(self,color)

    def set_image_texture(self,texture_map,inplace=True):

        mesh = self if inplace else copy.deepcopy(self)

        if isinstance(texture_map,str):
            texture_map = PIL.Image.open(texture_map)
            texture_map = np.array(texture_map,dtype=np.float) / 255.0
            texture_map = torch.tensor(texture_map,dtype=torch.float).to(device).permute(2,0,1).unsqueeze(0)


        mesh.texture_map = texture_map
        return mesh

    def export(self, file, color=None):
        with open(file, "w+") as f:
            for vi, v in enumerate(self.vertices):
                if color is None:
                    f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
                else:
                    f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
                if self.vertex_normals is not None:
                    f.write("vn %f %f %f\n" % (self.vertex_normals[vi, 0], self.vertex_normals[vi, 1], self.vertex_normals[vi, 2]))
            for face in self.faces:
                f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))

def get_texture_map_from_color(mesh, color, H=224, W=224):
    num_faces = mesh.faces.shape[0]
    texture_map = torch.zeros(1, H, W, 3).to(device)
    texture_map[:, :, :] = color
    return texture_map.permute(0, 3, 1, 2)


def get_face_attributes_from_color(mesh, color):
    num_faces = mesh.faces.shape[0]
    face_attributes = torch.zeros(1, num_faces, 3, 3).to(device)
    face_attributes[:, :, :] = color
    return face_attributes

def standardize_mesh(mesh):
    verts = mesh.vertices
    center = verts.mean(dim=0)
    verts -= center
    scale = torch.std(torch.norm(verts, p=2, dim=1))
    verts /= scale
    mesh.vertices = verts
    return mesh


def normalize_mesh(mesh):
    verts = mesh.vertices

    # Compute center of bounding box
    # center = torch.mean(torch.column_stack([torch.max(verts, dim=0)[0], torch.min(verts, dim=0)[0]]))
    center = verts.mean(dim=0)
    verts = verts - center
    scale = torch.max(torch.norm(verts, p=2, dim=1))
    verts = verts / scale
    mesh.vertices = verts
    return mesh