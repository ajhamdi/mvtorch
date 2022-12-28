import torch
from torch.autograd import Variable
import numpy as np
import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .utils import unit_spherical_grid, batch_tensor, class_freq_to_weight, labels2freq, torch_direction_vector, unbatch_tensor
from .ops import mvctosvc, svctomvc, batched_index_select_parts
from .models.pointnet import *
from torch import nn
from einops import rearrange



class CircularViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim = torch.linspace(-180, 180, self.nb_views+1)[:-1] - 90.0
        views_elev = torch.ones_like(
            views_azim, dtype=torch.float, requires_grad=False)*canonical_elevation
        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.5)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + \
                torch.normal(0.0, 180.0 * self.input_view_noise,
                             c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + \
                torch.normal(0.0, 90.0 * self.input_view_noise,
                             c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + \
                torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                             c_views_dist.size(), device=c_views_dist.device)
        return c_views_azim, c_views_elev, c_views_dist


class SphericalViewSelector(nn.Module):
    def __init__(self, nb_views=12,canonical_distance=2.2, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim, views_elev = unit_spherical_grid(self.nb_views)
        views_azim, views_elev = torch.from_numpy(views_azim).to(
            torch.float), torch.from_numpy(views_elev).to(torch.float)
        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.5)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + \
                torch.normal(0.0, 180.0 * self.input_view_noise,
                             c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + \
                torch.normal(0.0, 90.0 * self.input_view_noise,
                             c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + \
                torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                             c_views_dist.size(), device=c_views_dist.device)
        return c_views_azim, c_views_elev, c_views_dist


class RandomViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_distance=2.2,  transform_distance=False):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_elev = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        views_azim = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_azim = c_views_azim + \
            torch.rand((c_batch_size, self.nb_views),
                       device=c_views_azim.device) * 360.0 - 180.0
        c_views_elev = c_views_elev + \
            torch.rand((c_batch_size, self.nb_views),
                       device=c_views_elev.device) * 180.0 - 90.0
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.499)
        return c_views_azim, c_views_elev, c_views_dist


class LearnedDirectViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        views_elev = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        if self.transform_distance:
            self.view_transformer = Seq(MLP([shape_features_size, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 3*self.nb_views], dropout=0.5, norm=True), MLP([3*self.nb_views, 3*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())
        else:
            self.view_transformer = Seq(MLP([shape_features_size, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 2*self.nb_views], dropout=0.5, norm=True), MLP([2*self.nb_views, 2*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())

        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        if not self.transform_distance:
            adjutment_vector = self.view_transformer(shape_features)
            adjutment_vector = torch.chunk(adjutment_vector, 2, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0,  c_views_elev + adjutment_vector[1] * 89.9, c_views_dist
        else:
            adjutment_vector = self.view_transformer(shape_features)
            adjutment_vector = torch.chunk(adjutment_vector, 3, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0,  c_views_elev + adjutment_vector[1] * 89.9, c_views_dist + adjutment_vector[2] * c_views_dist + 0.1


class LearnedCircularViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim = torch.linspace(-180, 180, self.nb_views+1)[:-1]
        views_elev = torch.ones_like(
            views_azim, dtype=torch.float, requires_grad=False)*canonical_elevation
        if self.transform_distance:
            self.view_transformer = Seq(MLP([shape_features_size+3*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 3*self.nb_views], dropout=0.5, norm=True), MLP([3*self.nb_views, 3*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())
        else:
            self.view_transformer = Seq(MLP([shape_features_size+2*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 2*self.nb_views], dropout=0.5, norm=True), MLP([2*self.nb_views, 2*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())

        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + \
                torch.normal(0.0, 180.0 * self.input_view_noise,
                             c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + \
                torch.normal(0.0, 90.0 * self.input_view_noise,
                             c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                                                       c_views_dist.size(), device=c_views_dist.device)

        if not self.transform_distance:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 2, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist
        else:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev, c_views_dist], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 3, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist + adjutment_vector[2] * self.canonical_distance + 0.1


class LearnedSphericalViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim, views_elev = unit_spherical_grid(self.nb_views)
        views_azim, views_elev = torch.from_numpy(views_azim).to(
            torch.float), torch.from_numpy(views_elev).to(torch.float)
        if self.transform_distance:
            self.view_transformer = Seq(MLP([shape_features_size+3*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 3*self.nb_views], dropout=0.5, norm=True), MLP([3*self.nb_views, 3*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())
        else:
            self.view_transformer = Seq(MLP([shape_features_size+2*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 2*self.nb_views], dropout=0.5, norm=True), MLP([2*self.nb_views, 2*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())

        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.5)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + \
                torch.normal(0.0, 180.0 * self.input_view_noise,
                             c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + \
                torch.normal(0.0, 90.0 * self.input_view_noise,
                             c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + \
                torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                             c_views_dist.size(), device=c_views_dist.device)
        if not self.transform_distance:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 2, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist
        else:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev, c_views_dist], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 3, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist + adjutment_vector[2] * self.canonical_distance + 0.1


class LearnedRandomViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_distance=2.2, shape_features_size=512, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_elev = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        views_azim = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        if self.transform_distance:
            self.view_transformer = Seq(MLP([shape_features_size+3*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 3*self.nb_views], dropout=0.5, norm=True), MLP([3*self.nb_views, 3*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())
        else:
            self.view_transformer = Seq(MLP([shape_features_size+2*self.nb_views, shape_features_size, shape_features_size, 5 *
                                             self.nb_views, 2*self.nb_views], dropout=0.5, norm=True), MLP([2*self.nb_views, 2*self.nb_views], act=None, dropout=0, norm=False), nn.Tanh())

        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_azim = c_views_azim + \
            torch.rand((c_batch_size, self.nb_views),
                       device=c_views_azim.device) * 360.0 - 180.0
        c_views_elev = c_views_elev + \
            torch.rand((c_batch_size, self.nb_views),
                       device=c_views_elev.device) * 180.0 - 90.0
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.499)
        if not self.transform_distance:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 2, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist
        else:
            adjutment_vector = self.view_transformer(
                torch.cat([shape_features, c_views_azim, c_views_elev, c_views_dist], dim=1))
            adjutment_vector = torch.chunk(adjutment_vector, 3, dim=1)
            return c_views_azim + adjutment_vector[0] * 180.0/self.nb_views,  c_views_elev + adjutment_vector[1] * 90.0, c_views_dist + adjutment_vector[2] * self.canonical_distance + 0.1


class ViewSelector(nn.Module):
    def __init__(self, nb_views=12, views_config="circular", canonical_elevation=30.0, canonical_distance=2.2, shape_features_size=512, transform_distance=False, input_view_noise=0.0,):
        super().__init__()
        self.views_config = views_config
        self.nb_views = nb_views
        if self.views_config == "circular" or self.views_config == "custom" or (self.views_config == "spherical" and self.nb_views == 4):
            self.chosen_view_selector = CircularViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                                                             canonical_distance=canonical_distance, transform_distance=transform_distance, input_view_noise=input_view_noise)
        elif self.views_config == "spherical":
            self.chosen_view_selector = SphericalViewSelector(nb_views=nb_views,canonical_distance=canonical_distance,transform_distance=transform_distance, input_view_noise=input_view_noise)
        elif self.views_config == "random":
            self.chosen_view_selector = RandomViewSelector(nb_views=nb_views, canonical_distance=canonical_distance, transform_distance=transform_distance)
        elif self.views_config == "learned_circular" or (self.views_config == "learned_spherical" and self.nb_views == 4):
            self.chosen_view_selector = LearnedCircularViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                                                               canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance, input_view_noise=input_view_noise)
        elif self.views_config == "learned_direct":
            self.chosen_view_selector = LearnedDirectViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                                                               canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance)
        elif self.views_config == "learned_spherical":
            self.chosen_view_selector = LearnedSphericalViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                                                                  canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance)
        elif self.views_config == "learned_random":
            self.chosen_view_selector = LearnedRandomViewSelector(nb_views=nb_views, canonical_distance=canonical_distance, shape_features_size=shape_features_size, transform_distance=transform_distance, input_view_noise=input_view_noise)


    def forward(self, shape_features=None, c_batch_size=1):
        return self.chosen_view_selector(shape_features=shape_features, c_batch_size=c_batch_size)



class FeatureExtractor(nn.Module):
    def __init__(self,  shape_features_size, views_config, shape_extractor, screatch_feature_extractor=False):
        super().__init__()
        self.shape_features_size = shape_features_size
        if views_config == "circular" or views_config == "random" or views_config == "spherical" or views_config == "custom":
            self.features_origin = "zeros"
        else:
            self.features_origin = "points_features"
            if shape_extractor == "PointNet":
                self.fe_model = PointNet(40, alignment=True)
            elif shape_extractor == "DGCNN":
                self.fe_model = SimpleDGCNN(40)
            if not screatch_feature_extractor:
                print(shape_extractor)
                load_point_ckpt(self.fe_model,  shape_extractor,
                                ckpt_dir='./checkpoint')


    def forward(self, extra_info=None, c_batch_size=1):
        if self.features_origin == "zeros":
            return torch.zeros((c_batch_size, self.shape_features_size))
        elif self.features_origin == "points_features":
            extra_info = extra_info.transpose(1, 2).to(
                next(self.fe_model.parameters()).device)
            features = self.fe_model(extra_info)

            return features[0].view(c_batch_size, -1)


class MVTN(nn.Module):
    """
    The MVTN main class that includes two components. one that extracts features from the object and one that predicts the views and other rendering setup. It is trained jointly with the main multi-view network.
    Args: 
        `nb_views` int , The number of views used in the multi-view setup
        `views_config`: str , The type of view selection method used. Choices: ["circular", "random", "learned_circular", "learned_direct", "spherical", "learned_spherical", "learned_random", "learned_transfer", "custom"]  
        `canonical_elevation`: float , the standard elevation of the camera viewpoints (if `views_config` == circulart).
        `canonical_distance`: float , the standard distance to the object of the camera viewpoints.
        `transform_distance`: bool , flag to allow for distance transformations from 0.5 `canonical_distance` to 1.5 `canonical_distance`
        `input_view_noise` : bool , flag to allow for adding noise to the camera viewpoints positions
        `shape_extractor` : str , The type of network used to extract features necessary for MVTN. Choices: ["PointNet", "DGCNN",]
        `shape_features_size`: float , the features size extracted used in MVTN. It depends on the `shape_extractor` used 
        `screatch_feature_extractor` : bool , flag to not use pretrained weights for the `shape_extractor`. default is to use the pretrinaed weights on ModelNet40
    Returns:
        an MVTN object that can render multiple views according to predefined setup
    """

    def __init__(self, nb_views=12, views_config="circular", canonical_elevation=30.0, canonical_distance=1.1, transform_distance=False, input_view_noise=0.0, shape_extractor="pointnet", shape_features_size=512, screatch_feature_extractor=False):
        super().__init__()
        self.view_selector = ViewSelector(nb_views=nb_views, views_config=views_config, canonical_elevation=canonical_elevation, canonical_distance=canonical_distance,
                                          shape_features_size=shape_features_size, transform_distance=transform_distance, input_view_noise=input_view_noise,)
        self.feature_extractor = FeatureExtractor(shape_features_size=shape_features_size, views_config=views_config,
                                                  shape_extractor=shape_extractor, screatch_feature_extractor=screatch_feature_extractor)


    def forward(self, points=None, c_batch_size=1):
        shape_features = self.feature_extractor(points, c_batch_size)
        return self.view_selector(shape_features=shape_features, c_batch_size=c_batch_size)


        #

    def load_mvtn(self,weights_file):
        # Load checkpoint.
        print('\n==> Loading checkpoint..')
        assert os.path.isfile(weights_file
                            ), 'Error: no checkpoint file found!'
        checkpoint = torch.load(weights_file)
        self.load_state_dict(checkpoint['mvtn'])

def batch_objectclasses2weights(batch_classes, cls_freq,alpha=1.0):
    cls_freq = {k: v for k, v in cls_freq}
    class_weight = class_freq_to_weight(cls_freq, alpha)
    class_weight = [class_weight[x] for x in sorted(class_weight.keys())]
    c_batch_weights = torch.ones_like(
        batch_classes).cuda() * torch.Tensor(class_weight).cuda()[batch_classes.cpu().numpy().tolist()]
    return c_batch_weights


def batch_segmentclasses2weights(batch_seg_classes, alpha=0.0):
    c_bs = batch_seg_classes.shape[0]
    c_batch_weights = torch.ones_like(batch_seg_classes).to(batch_seg_classes.device)
    if alpha >0.0 :
        for ii in range(c_bs):
            cls_freq, indx_map = labels2freq(batch_seg_classes[ii])
            class_weight = class_freq_to_weight(cls_freq, alpha)
            class_weight = torch.Tensor([class_weight[x]
                                        for x in sorted(class_weight.keys())]).cuda()

            # c_batch_weights[ii] * torch.Tensor(class_weight)[batch_seg_classes[ii].cpu().numpy().tolist()]
            c_batch_weights[ii] = torch.gather(class_weight, dim=0, index=indx_map.view(-1)).view(c_batch_weights[ii].size())
    return c_batch_weights

class MVPartSegmentation(nn.Module):
    def __init__(self, model, num_classes, num_parts, parallel_head=True, balanced_object_loss=True, balanced_2d_loss_alpha=0.0, depth=2, total_nb_parts=50):
        super().__init__()
        self.num_classes = num_classes
        self.model = model
        self.cls_freq = [('Airplane', 341), ('Bag', 14), ('Cap', 11), ('Car', 158), ('Chair', 704), ('Earphone', 14), ('Guitar', 159), ('Knife', 80),
                    ('Lamp', 286), ('Laptop', 83), ('Motorbike', 51), ('Mug', 38), ('Pistol', 44), ('Rocket', 12), ('Skateboard', 31), ('Table', 848)]  # classes frequency of ShapeNEts parts 
        self.multi_shape_heads = nn.ModuleList()
        self.parallel_head = parallel_head
        self.balanced_object_loss = balanced_object_loss
        self.balanced_2d_loss_alpha = balanced_2d_loss_alpha
        if parallel_head:
            for _ in range(num_classes):
                layers = [torch.nn.Conv2d(21, 2*num_parts, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(2*num_parts),
                          nn.ReLU(inplace=True)] + [torch.nn.Conv2d(2*num_parts, 2*num_parts, kernel_size=(1, 1), stride=(1, 1)),
                                                    nn.BatchNorm2d(2*num_parts),nn.ReLU(inplace=True)] * int(depth - 2) + [torch.nn.Conv2d(2*num_parts, num_parts+1, kernel_size=(1, 1), stride=(1, 1))
                ]
                self.multi_shape_heads.append(nn.Sequential(*layers))
        else:
            layers =  [torch.nn.Conv2d(21, 21, kernel_size=(1, 1), stride=(1, 1)),
                                                        nn.BatchNorm2d(21),
                       nn.ReLU(inplace=True)] * int(depth - 1) + [torch.nn.Conv2d(21, total_nb_parts+1,
                                                                        kernel_size=(1, 1), stride=(1, 1))
            ]
            self.multi_shape_heads.append(nn.Sequential(*layers))


    def forward(self, mvimages, cls):
        bs,nb_views, _,h,w = mvimages.shape
        features = self.model(mvctosvc(mvimages))["out"]
        if self.parallel_head:
            logits_all_shapes = []
            for ii in range(self.num_classes):
                logits_all_shapes.append(
                    self.multi_shape_heads[ii](features)[..., None])
            outputs = torch.cat(logits_all_shapes, dim=4) #######

            target_mask = torch.arange(0, self.num_classes)[None, ...].repeat(bs, 1).cuda() == cls
            predict_mask = target_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(torch.float).repeat(1, nb_views, 1, 1, 1, 1)
            outputs = torch.max(outputs * rearrange(predict_mask, 'b V C h w cls -> (b V) C h w cls'), dim=-1)[0]

        else:
            outputs = self.multi_shape_heads[0](features)
        return outputs, features

    def get_loss(self, criterion, outputs, labels_2d,cls):
        nb_views= labels_2d.shape[1]
        # loss = criterion(outputs, mvtosv(labels_2d.to(torch.long)))
        loss = criterion(svctomvc(outputs, nb_views=nb_views).transpose(1,2),labels_2d.to(torch.long))
        if self.training:  
            loss_compensation = 300 * (1-float(self.balanced_object_loss)) + 1
            loss_factor_object = batch_objectclasses2weights(cls.squeeze(), self.cls_freq, alpha=float(self.balanced_object_loss))
            # loss_factor_object = torch.repeat_interleave(loss_factor_object, (nb_views))[..., None][..., None][..., None]
            loss_factor_object = loss_factor_object[...,None][..., None][..., None]

            loss_factor_segment = batch_segmentclasses2weights(labels_2d, alpha=self.balanced_2d_loss_alpha)
            
            loss = loss_compensation * (loss * loss_factor_object * loss_factor_segment).mean()

        return loss

class MVLiftingModule(nn.Module):
    def __init__(self, image_size, lifting_method="mode", lifting_net=None, mlp_classifier=None, balanced_object_loss=False, balanced_3d_loss_alpha=0.0, use_early_voint_feats=False):
        super().__init__()
        self.image_size = image_size
        self.lifting_method = lifting_method
        self.use_early_voint_feats = use_early_voint_feats
        self.lifting_net = lifting_net
        self.balanced_object_loss = balanced_object_loss
        self.balanced_3d_loss_alpha = balanced_3d_loss_alpha
        self.mlp_classifier = mlp_classifier
        self.cls_freq = [('Airplane', 341), ('Bag', 14), ('Cap', 11), ('Car', 158), ('Chair', 704), ('Earphone', 14), ('Guitar', 159), ('Knife', 80),
                         ('Lamp', 286), ('Laptop', 83), ('Motorbike', 51), ('Mug', 38), ('Pistol', 44), ('Rocket', 12), ('Skateboard', 31), ('Table', 848)]  # classes frequency of ShapeNEts parts
    def forward(self,):
        pass

    def compute_views_weights(self, azim,elev, rendered_images, normals):
        c_batch_size = azim.shape[0]
        dir_vec = torch_direction_vector(batch_tensor(azim, dim=1, squeeze=True), batch_tensor(elev, dim=1, squeeze=True))
        dir_vec = unbatch_tensor(dir_vec, dim=1, unsqueeze=True, batch_size=c_batch_size)

        if "attention" in self.lifting_method:
            if self.lifting_method == "point_attention":
                points_weights = torch.abs(
                    torch.bmm(dir_vec.cuda(), normals.transpose(1, 2).cuda()))
                views_weights = points_weights.sum(dim=-1)
                views_weights = views_weights / views_weights.sum(dim=-1, keepdim=True)
                views_weights = views_weights[...,
                    None][..., None][..., None]
            elif self.lifting_method == "pixel_attention":
                normal_weight_maps = dir_vec.cuda()[..., None][..., None].repeat(
                    1, 1, 1, self.image_size, self.image_size) * rendered_images
                views_weights = torch.abs(
                    normal_weight_maps.sum(dim=2, keepdim=True))
            elif self.lifting_method == "view_attention":
                normal_weight_maps = dir_vec.cuda()[..., None][..., None].repeat(
                    1, 1, 1, self.image_size, self.image_size) * rendered_images
                views_weights = torch.abs(normal_weight_maps.sum(dim=2, keepdim=True)).sum(
                    dim=3, keepdim=True).sum(dim=4, keepdim=True)
                views_weights = views_weights / views_weights.sum(dim=1, keepdim=True)
        else :
            views_weights = torch.ones_like(azim).cuda()[..., None][..., None][..., None]
        return views_weights
            
    def compute_image_segment_label_points(self,points, batch_points_labels, rendered_pix_to_point,):
        """
        Compute ground truth segmentation labels for rendered images.
        Args:
            args: bunch of arguments.
            rendered_pix_to_point: (B * nb_view, h, w).
            batch_meshes: list[Mesh], with length of B
            device: device string
            batch_points_labels: Tensor with shape (B, max_face_label_length), padded with -1.
        Returns:
            labels_2d: (B * nb_view, h, w).
            pix_to_face_mask: (B * nb_view, h, w). Valid mask for labels_2d.
        """
        # invalid value in pix_to_face is -1, use it to build a mask.
        # (batch_size * nb_view, h, w)
        ### adjustment
        B, N, _ = points.shape
        pix_to_face_mask = ~rendered_pix_to_point.eq(-1)
        rendered_pix_to_point = rendered_pix_to_point % N + 1
        rendered_pix_to_point[~pix_to_face_mask] = 0
        batch_points_labels = torch.cat((torch.zeros(B)[..., None].to(
            torch.int32).cuda(), batch_points_labels), dim=1)

        # unpack pix_to_face
        class_map = batched_index_select_parts(batch_points_labels[:, None, ...], rendered_pix_to_point)
        # take only max point label for each pixel
        labels_2d = class_map[:, :, 0, 0, ...]

        return labels_2d, pix_to_face_mask

    def lift_2D_to_3D(self, points, predictions_2d, rendered_pix_to_point, views_weights,  cls, parts_nb,view_info=None,early_feats= None):
        """
        Unproject the 2d predictions of segmentation labels to 3d.
        Args:
            rendered_pix_to_point: (B * nb_view, h, w).
            device: device string
        Returns:
            labels_3d: (B * nb_points, ).
        """
        B, N, _ = points.shape
        nb_views, nb_classes = predictions_2d.shape[1], predictions_2d.shape[2]
        pix_to_face_mask = ~rendered_pix_to_point.eq(-1)
        rendered_pix_to_point = rendered_pix_to_point % N
        rendered_pix_to_point[~pix_to_face_mask] = 0

        predictions_2d = torch.nn.functional.softmax(predictions_2d, dim=2)
        _, predictions_2d_max = torch.max(predictions_2d.detach().clone(), dim=2)
        # predictions_2d_max = predictions_2d
        if self.use_early_voint_feats:
            nb_classes = early_feats.shape[1]
            predictions_2d = views_weights * svctomvc(early_feats, nb_views=nb_views)
        else:
            predictions_2d = views_weights * predictions_2d

        labels_3d = torch.zeros((B, N, nb_classes,)).cuda()
        labels_count = torch.zeros((B, N, nb_classes), dtype=torch.long).cuda()
        voints_features = torch.zeros(
            (B, N, nb_classes, nb_views), dtype=torch.float).cuda()
        voints_mask = torch.zeros((B, N, nb_views), dtype=torch.float).cuda()

        if self.lifting_method == "mode":
            ########################################### 
            for batch in range(B):
                for ii, part in enumerate(range(1, 1 + parts_nb[batch].item())):
                    points_indices = rendered_pix_to_point[batch][predictions_2d_max[:, :, None, ...].expand(
                        rendered_pix_to_point.size())[batch] == part]

                    points_indices, points_counts = torch.unique(
                        points_indices, return_counts=True)
                    labels_count[batch, :, ii][points_indices] = points_counts
                    labels_3d[batch, :, ii][points_indices] = part
            _, indcs = torch.max(labels_count, dim=-1)
            empty_points = torch.sum(labels_3d, dim=-1) == 0
            labels_3d = indcs + 1
            labels_3d[empty_points] = 0
            labels_3d_feats = torch.nn.functional.one_hot(labels_3d, num_classes=nb_classes).to(torch.float).transpose(1, 2)
        # elif self.lifting_method == "mean" or "attention" in self.lifting_method or self.lifting_method == "max":
        else:
            # labels_3d_feats = torch.zeros((B, N, nb_classes,)).cuda()
            for batch in range(B):
                for view in range(nb_views):
                    # selected_feats = []
                    for ii, part in enumerate(range(1, 1 + parts_nb[batch].item())):
                        class_label_mask = predictions_2d_max[:, :, None, ...].expand(rendered_pix_to_point.size())[batch,view,...] == part
                        points_indices = rendered_pix_to_point[batch,view, ...][class_label_mask]
                        labels_3d[batch, :, ii][points_indices] = part
                        voints_mask[batch, :, view][points_indices] = 1 

                        index_list = torch.arange(len(class_label_mask.squeeze().view(-1)))
                        index_tensor = index_list[class_label_mask.squeeze().view(-1)][None, ...].repeat(nb_classes, 1).cuda()
                        selected_feats = torch.gather(predictions_2d[batch, view, ...].view(nb_classes,-1), dim=1, index=index_tensor)

                        voints_features[batch,:,:,view][points_indices,:] = selected_feats.transpose(0,1)

            voints_features = voints_features.transpose(1, 2)
            # empty_points = torch.sum(voints_mask, dim=-1) == 0
            if self.lifting_method == "mean" or "attention" in self.lifting_method:
                labels_3d_feats = vointmean(voints_features, voints_mask)
            elif self.lifting_method == "max":
                labels_3d_feats = vointmax(voints_features, voints_mask)
            elif self.lifting_method in ["mlp","gcn","transformer","gat"]:
                labels_3d_feats = self.lifting_net(voints_features, voints_mask, view_info, points.transpose(1, 2))

        return self.mlp_classifier(labels_3d_feats, points.transpose(1, 2),cls)
# 

    def get_loss_3d(self, criterion, predictions_3d, labels_3d, cls):
        loss = criterion(predictions_3d, labels_3d.to(torch.long))
        if self.training:
            # to compnsate for the reduced loss due to weighting
            loss_compensation = 300 * (1-float(self.balanced_object_loss)) + 1
            loss_factor_object = batch_objectclasses2weights(cls.squeeze(), self.cls_freq, alpha=float(self.balanced_object_loss))
            loss_factor_segment = batch_segmentclasses2weights(labels_3d.squeeze(), alpha=self.balanced_3d_loss_alpha)
            loss = loss_compensation * (loss * loss_factor_object[..., None]* loss_factor_segment).mean()


        return loss