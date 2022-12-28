# to import files from parent dir
# import sys
# import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ops import mvctosvc, batched_index_select_parts, svctomvc
from .utils import batch_tensor, unbatch_tensor, class_freq_to_weight, torch_direction_vector, labels2freq
import torch
import numpy as np
import torchvision
from torch import dropout, nn
# from .models.blocks import act_layer, Conv1dLayer , MLP
from mvtorch.models.voint import *

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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


class ViewMaxAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, M, C, H, W = mvimages.shape
        pooled_view = torch.max(self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)), dim=1)[0]
        return pooled_view.squeeze()


class ViewAvgAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, M, C, H, W = mvimages.shape
        pooled_view = torch.mean(self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True)), dim=1)
        return pooled_view.squeeze()




class MVAggregate(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=512, num_classes=1000,lifting_net=nn.Sequential()):
        super().__init__()
        self.agr_type = agr_type

        self.fc = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, num_classes)
        )
        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAggregate(model=model, lifting_net=lifting_net)

    def forward(self, mvimages):
        pooled_view = self.aggregation_model(mvimages)
        predictions = self.fc(pooled_view)
        return predictions, pooled_view





class FullCrossViewAttention(nn.Module):
    def __init__(self, model,  patch_size=16, num_views=1, feat_dim=512, num_classes=1000):
        super().__init__()
        self.model = model
        self.model.pos_embed = nn.Parameter(torch.cat(
            (self.model.pos_embed[:, 0, :].unsqueeze(1), self.model.pos_embed[:, 1::, :].repeat((1, num_views, 1))), dim=1))
        # self.model.pos_embed.retain_grad()
        self.combine_views = Rearrange(
            'b N c (h p1) (w p2) -> b c (h p1 N) (w p2)', p1=patch_size, p2=patch_size, N=num_views)
        self.fc = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, mvimages):
        mvimages = self.combine_views(mvimages)
        feats = self.model(mvimages)
        return self.fc(feats), feats


class WindowCrossViewAttention(nn.Module):
    def __init__(self, model,  patch_size=16, num_views=1, num_windows=1, feat_dim=512, num_classes=1000, agr_type="max"):
        super().__init__()

        assert num_views % num_windows == 0, "the number of winsows should be devidand of number of views "
        view_per_window = int(num_views/num_windows)
        model.pos_embed = nn.Parameter(torch.cat((model.pos_embed[:, 0, :].unsqueeze(
            1), model.pos_embed[:, 1::, :].repeat((1, view_per_window, 1))), dim=1))
        self.model = MVAggregate(model, agr_type=agr_type,
                                feat_dim=feat_dim, num_classes=num_classes)
        self.combine_views = Rearrange('b (Win NV) c (h p1) (w p2) -> b Win c (h p1 NV) (w p2)',
                                       p1=patch_size, p2=patch_size, Win=num_windows, NV=view_per_window)

    def forward(self, mvimages):
        mvimages = self.combine_views(mvimages)
        pred, feats = self.model(mvimages)
        return pred, feats



class MVPartSegmentation(nn.Module):
    def __init__(self,  model, num_classes, num_parts, parallel_head=False, balanced_object_loss=True, balanced_2d_loss_alpha=0.0, depth=2,total_nb_parts=50):
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
