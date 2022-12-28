# to import files from parent dir
import sys
import os

import torch
import numpy as np
from torch import nn, einsum, dropout
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .blocks import *
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..utils import positional_encoding


class PointMLPClassifier(nn.Module):
    def __init__(self, in_size, out_size, feat_dim=64, use_global=False, use_xyz=False, skip=False, parallel_head=False,nb_heads=1,extra_net=None):
        super().__init__()
        self.extra_net = extra_net
        if use_xyz :
            in_size += 3 
        self.feature_extractor = Conv1dLayer([in_size, feat_dim, feat_dim], act='relu', norm=True, bias=True)
        if use_global:
            self.aux_classifier = MLP([feat_dim, feat_dim, feat_dim])
            self.classifier = nn.Sequential(Conv1dLayer(
                [feat_dim*2, feat_dim], act='relu', norm=True, bias=True), Conv1dLayer([feat_dim, out_size], act='relu', norm=True, bias=False))
        else:
            self.aux_classifier = nn.Sequential()
            self.classifier = Conv1dLayer([feat_dim, out_size], act='relu', norm=True, bias=False)

        self.use_global = use_global
        self.use_xyz = use_xyz 
        self.skip = skip
        self.parallel_head = parallel_head
        self.nb_heads = nb_heads
        if parallel_head:
            self.multi_shape_heads = nn.ModuleList()
            for _ in range(nb_heads):
                self.multi_shape_heads.append(nn.Sequential(
                    Conv1dLayer([out_size, out_size, out_size], act='relu',
                                norm=True, bias=False)
                ))

    def forward(self, points_feats, xyz=None, cls=None):
        bs ,_ , nb_points = points_feats.shape
        if self.skip:
            return points_feats
        if self.use_xyz:
            points_feats = torch.cat((xyz, points_feats),dim=1)
        points_feats = self.feature_extractor(points_feats)
        if self.use_global:
            global_points_feats = self.aux_classifier(torch.max(points_feats,dim=-1)[0])
            points_feats = torch.cat((global_points_feats[..., None].repeat(1,1, nb_points), points_feats), dim=1)
        if not self.parallel_head:
            return self.extra_net(self.classifier(points_feats))
        else : 
            points_feats = self.extra_net(self.classifier(points_feats))
            logits_all_shapes = []
            for ii in range(self.nb_heads):
                logits_all_shapes.append(
                    self.multi_shape_heads[ii](points_feats)[..., None])
            outputs = torch.cat(logits_all_shapes, dim=3)
            # out_size = outputs.shape[1]
            # feats = torch.gather(outputs, dim=3, index=cls.unsqueeze(1).unsqueeze(1).repeat(1, out_size, nb_points, 1).to(torch.long)).squeeze(-1)
            target_mask = torch.arange(0, self.nb_heads)[None, ...].repeat(bs, 1).cuda() == cls
            predict_mask = target_mask.unsqueeze(1).unsqueeze(1).to(torch.float)
            feats = torch.max(outputs * predict_mask, dim=-1)[0]
            return feats 


def reset_voint_empty(reduced_voints_feats, voints_mask):
    empty_voints = torch.sum(voints_mask, dim=-1) == 0
    reduced_voints_feats.transpose(1, 2)[empty_voints] = 0.0
    return reduced_voints_feats


def reset_voint_mask(voints_feats, voints_mask):
    voints_feats = voints_feats * voints_mask[:, None, ...]
    return voints_feats


def vointgraphmax(voints_feats, voints_mask, return_indices=False, dim=-1):
        voints_feats, voints_inidices = torch.max(
            voints_feats - 1e10 * (~ voints_mask[:, None, ...].to(torch.bool)).to(torch.float), dim=dim)
        voints_feats = reset_voint_mask(voints_feats, voints_mask.squeeze(-1))
        if return_indices:
            return voints_feats, voints_inidices
        else:
            return voints_feats


def vointgraphmean(voints_feats, voints_mask, dim=-1):
        voints_cardinality = voints_mask.sum(dim=dim)
        voints_feats = voints_feats.sum(
            dim=dim)/voints_cardinality[:, None, ...]
        voints_feats = reset_voint_mask(voints_feats, voints_mask.squeeze(-1))
        return voints_feats


def vointattentionpool(voints_feats, voints_mask,attention_wights, dim=-1):
        voints_cardinality = voints_mask.sum(dim=dim)
        voints_feats =torch.sum(attention_wights*voints_feats,dim=dim)/voints_cardinality[:, None, ...]
        voints_feats = reset_voint_mask(voints_feats, voints_mask.squeeze(-1))
        return voints_feats

def vointmax(voints_feats, voints_mask,return_indices=False,dim=-1):
        voints_feats, voints_inidices = torch.max(voints_feats - 1e10 * (~ voints_mask[:, None, ...].to(torch.bool)).to(torch.float), dim=dim)
        voints_feats = reset_voint_empty(voints_feats, voints_mask)
        if return_indices:
            return voints_feats, voints_inidices
        else :
            return voints_feats


def vointmean(voints_feats, voints_mask,dim=-1):
        voints_cardinality = voints_mask.sum(dim=dim)
        voints_feats = voints_feats.sum(dim=dim)/voints_cardinality[:, None, ...]
        voints_feats = reset_voint_empty(voints_feats, voints_mask)
        return voints_feats

def vointsoftmax(voints_feats, voints_mask,dim=-1):
        voints_feats = torch.nn.functional.softmax(voints_feats - 1e10 * (~ voints_mask[:, None, ...].to(torch.bool)).to(torch.float), dim=dim)
        voints_feats = reset_voint_mask(voints_feats, voints_mask)
        return voints_feats


class ViewEmbedding(nn.Module):
    def __init__(self, view_embeddgin_type="none", embed_dim=24, use_view_info=False, nb_view_angles=2):
        super().__init__()
        self.view_embeddgin_type = view_embeddgin_type
        self.nb_view_angles = nb_view_angles  # the azimuth and elevation
        self.embed_dim = embed_dim
        if self.embed_dim < 4:
            self.view_embeddgin_type = "none"
        self.use_view_info = use_view_info
        if view_embeddgin_type == "learned":
            pass  # TODO

    def forward(self, view_info=None):
        if not self.use_view_info:
            return None
        else:
            if self.view_embeddgin_type == "none":
                return view_info
            elif self.view_embeddgin_type == "zeros":
                return torch.zeros_like(view_info)
            elif self.view_embeddgin_type == "sin":
                return positional_encoding(view_info, num_encoding_functions=int(self.embed_dim/4), include_input=False, log_sampling=True)
            elif self.view_embeddgin_type == "learned":
                pass  # TODO
            elif self.view_embeddgin_type == "fourier":
                pass  # TODO
            

    def added_dim(self,):
        if not self.use_view_info:
            return 0
        elif self.view_embeddgin_type == "none" or self.view_embeddgin_type == "zeros":
            return self.nb_view_angles
        else:
            return self.embed_dim

class VointConv(nn.Module):
    def __init__(self, channels, act='relu', norm=True, bias=False, kernel_size=1, stride=1, dilation=1, drop=0., groups=1):
        super().__init__()
        self.layers  = nn.ModuleList()
        for i in range(1, len(channels)):
            self.layers.append(nn.Conv2d(channels[i - 1], channels[i], bias=bias,kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups))
            if norm:
                self.layers.append(nn.BatchNorm2d(channels[i]))
            if act:
                self.layers.append(act_layer(act))
            if drop > 0:
                self.layers.append(nn.Dropout2d(drop))


    def forward(self,voints_feats, voints_mask):
        for m in self.layers:
            voints_feats = m(voints_feats)
            voints_feats = reset_voint_mask(voints_feats, voints_mask)
        return voints_feats


class VointGraphConv(nn.Module):
    def __init__(self, channels, act='relu', norm=True, bias=False, kernel_size=1, stride=1, dilation=1, drop=0., groups=1, aggr="max"):
        super().__init__()
        if aggr == "max":
            self.aggr = vointgraphmax
        elif aggr == "mean":
            self.aggr = vointgraphmean
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            ml = []
            ml.append(nn.Conv3d(channels[i - 1], channels[i], bias=bias,
                                         kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups))
            if norm:
                ml.append(nn.BatchNorm3d(channels[i]))
            if act:
                ml.append(act_layer(act))
            if drop > 0:
                ml.append(nn.Dropout3d(drop))
            self.layers.append(nn.Sequential(*ml))

        self.embed_layers = nn.ModuleList()
        for i in range(1, len(channels)):
            self.embed_layers.append(VointConv([channels[i - 1],int(0.5* channels[i - 1])], act=act, norm=norm, bias=True, drop=drop))


    def forward(self, voints_feats, voints_mask):
        voints_mask = voints_mask[...,None]
        for ii, m in enumerate(self.layers):
            super_voints_feats = self.embed_layers[ii](voints_feats, voints_mask.squeeze(-1))
            bs, C, nb_voints, nb_views = super_voints_feats.shape

            nbrs_voints_feats = torch.zeros_like(super_voints_feats)[..., None].repeat(1, 1, 1, 1, nb_views).cuda()
            for jj in range(0, nb_views):
                nbrs_voints_feats[:,:,:,:,jj] = torch.roll(super_voints_feats, jj, 3).detach().clone() # nbrs gradients are not important 
            nbrs_voints_feats = nbrs_voints_feats - super_voints_feats[..., None]
            super_voints_feats = torch.cat((super_voints_feats[..., None].repeat(1, 1, 1, 1, nb_views), nbrs_voints_feats), dim=1)
            super_voints_feats = m(super_voints_feats)
            super_voints_feats = reset_voint_mask(super_voints_feats, voints_mask)
            voints_feats = voints_feats +  self.aggr(super_voints_feats, voints_mask)
        return voints_feats


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
class VointGraphAttention(nn.Module):
    def __init__(self, channels, act='relu', norm=True, bias=False, kernel_size=1, stride=1, dilation=1,aggr="max", drop=0., groups=1):
        super().__init__()
        if aggr == "max":
            self.aggr = vointgraphmax
        elif aggr == "mean":
            self.aggr = vointgraphmean
        self.atten_aggr = vointattentionpool
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            ml = []
            ml.append(nn.Conv3d(channels[i - 1], channels[i], bias=bias,
                                kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups))
            if norm:
                ml.append(nn.BatchNorm3d(channels[i]))
            if act:
                ml.append(act_layer(act))
            if drop > 0:
                ml.append(nn.Dropout3d(drop))
            self.layers.append(nn.Sequential(*ml))

        self.atten_layers = nn.ModuleList()
        for i in range(1, len(channels)):
            ml = []
            ml.append(nn.Conv3d(channels[i - 1], channels[i], bias=bias,
                                kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups))
            if norm:
                ml.append(nn.BatchNorm3d(channels[i]))
            if act:
                ml.append(act_layer(act))
            if drop > 0:
                ml.append(nn.Dropout3d(drop))
            self.atten_layers.append(nn.Sequential(*ml))

        self.embed_layers = nn.ModuleList()
        for i in range(1, len(channels)):
            self.embed_layers.append(VointConv(
                [channels[i - 1], int(channels[i - 1])], act=act, norm=norm, bias=True, drop=drop))
        self.embed_atten_layers = nn.ModuleList()
        for i in range(1, len(channels)):
            self.embed_atten_layers.append(VointConv(
                [channels[i - 1], int(channels[i - 1])], act=act, norm=norm, bias=True, drop=drop))

    def forward(self, voints_feats, voints_mask):
        bs, C, nb_voints, self.nb_views = voints_feats.shape
        voints_mask = voints_mask[..., None]
        for ii, _ in enumerate(self.layers):
            atten_cents, atten_nbrs = self.get_nbr_center_voints(voints_feats, voints_mask.squeeze(-1), self.embed_atten_layers[ii])
            main_cents, main_nbrs = self.get_nbr_center_voints(voints_feats, voints_mask.squeeze(-1), self.embed_layers[ii])
            atten_cents = self.apply_voint_graph_conv( atten_cents, atten_nbrs, voints_mask, self.atten_layers[ii])
            main_cents = self.apply_voint_graph_conv(main_cents, main_nbrs, voints_mask, self.layers[ii])

            atten_ws = vointsoftmax(self.aggr(atten_cents, voints_mask, dim=-2), voints_mask.squeeze(-1), dim=-1).unsqueeze(-2).cuda()
            
            voints_feats = voints_feats + self.atten_aggr(main_cents, voints_mask, atten_ws,dim=-2)

        return voints_feats

    def get_nbr_center_voints(self,voints_feats, voints_mask,embed_layer):
        super_voints_feats = embed_layer(voints_feats, voints_mask)
        super_voints_feats, nbrs_voints_feats = torch.chunk(super_voints_feats,2,dim=1)
        nbrs_voints_feats = nbrs_voints_feats[..., None].repeat(1, 1, 1, 1, self.nb_views)
        for jj in range(0, self.nb_views):
            nbrs_voints_feats[:, :, :, :, jj] = torch.roll(nbrs_voints_feats[..., 0], jj, 3) #.detach().clone()  # nbrs gradients are not important
        return super_voints_feats, nbrs_voints_feats

    def apply_voint_graph_conv(self, voints_cents, voints_nbrs, voints_mask, conv_layer):
            voints_cents = torch.cat((voints_cents[..., None].repeat(1, 1, 1, 1, self.nb_views), voints_nbrs), dim=1)
            voints_cents = conv_layer(voints_cents)
            voints_cents = reset_voint_mask(voints_cents, voints_mask)
            return voints_cents
class VointSelfAttention(nn.Module):
    def __init__(self, channels, act='relu', norm=True, bias=False, kernel_size=1, stride=1, dilation=1, aggr="max", drop=0., groups=1,heads=4):
        super().__init__()
        if aggr == "max":
            self.aggr = vointgraphmax
        elif aggr == "mean":
            self.aggr = vointgraphmean
        self.atten_aggr = vointattentionpool
        self.heads = heads

        self.feedforward = nn.ModuleList()
        for i in range(1, len(channels)):
            self.feedforward.append(VointConv([channels[i - 1], int(channels[i - 1])], act=act, norm=norm, bias=True, drop=drop))

        self.to_out = nn.ModuleList()
        for i in range(1, len(channels)):
            self.to_out.append(VointConv(
                [channels[i - 1], int(channels[i - 1])], act=act, norm=norm, bias=True, drop=drop))

        self.embed_layers = nn.ModuleList()
        for i in range(1, len(channels)):
            self.embed_layers.append(VointConv(
                [channels[i - 1], int(channels[i - 1]/self.heads)], act=act, norm=norm, bias=True, drop=drop))
        self.embed_atten_layers = nn.ModuleList()
        self.scale = []

        for i in range(1, len(channels)):
            self.embed_atten_layers.append(VointConv([int(channels[i - 1]/self.heads), 3*int(channels[i - 1])], act=act, norm=norm, bias=True, drop=drop))
            self.scale.append(float(channels[i - 1]) ** -0.5)

    def forward(self, voints_feats, voints_mask):
        bs, C, nb_voints, self.nb_views = voints_feats.shape
        # voints_mask = voints_mask[..., None]
        for ii, _ in enumerate(self.embed_atten_layers):
            embedded_voints = self.embed_layers[ii](voints_feats, voints_mask)
            voint_k, voint_q, voint_v = self.get_qkv(embedded_voints, voints_mask, self.embed_atten_layers[ii])
            dots = einsum('b h d n v, b h d n j-> b h n v j',voint_q, voint_k) * self.scale[ii]
            dots = reset_voint_empty(dots, voints_mask)

            attn = vointsoftmax(dots,voints_mask[...,None], dim=-1)

            out = einsum('b h n v j, b h d n v-> b h n v d', attn, voint_v)
            out = rearrange(out, 'b h n v d -> b (h d) n v')
            out = reset_voint_empty(out, voints_mask)
            out = self.to_out[ii](out, voints_mask)
            voints_feats = voints_feats + out
            voints_feats = voints_feats + self.feedforward[ii](voints_feats, voints_mask)

        return voints_feats

    def get_qkv(self, voints_feats, voints_mask, embed_layer):
        super_voints_feats = embed_layer(voints_feats, voints_mask)
        qkv = torch.chunk(super_voints_feats, 3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n v -> b h d n v', h=self.heads), qkv)
        return q, k, v


class VointMLP(nn.Module):
    def __init__(self, in_size, out_size, feat_dim=64, use_cls_voint=False, viewembedder=None, aggr="max", use_xyz=False, voint_depth=2):
        super().__init__()
        self.use_cls_voint = use_cls_voint
        self.use_xyz = use_xyz
        if use_xyz:
            in_size += 3 
        if aggr == "max":
            self.aggr = vointmax
        elif aggr == "mean":
            self.aggr = vointmean
        self.view_embed = viewembedder
        in_size += self.view_embed.added_dim()
        self.feature_extractor = VointConv(
            [in_size] + [feat_dim] * int(voint_depth+1), act='relu', norm=True, bias=True, drop=0.0)
        if use_cls_voint:
            self.aux_classifier = VointConv([feat_dim, feat_dim, feat_dim], act='relu', norm=True, bias=True)
            self.classifier = VointConv([feat_dim*2,feat_dim, out_size], act='relu', norm=True, bias=False)
        else:
            self.aux_classifier = nn.Sequential()
            self.classifier = VointConv(
                [feat_dim, out_size], act='relu', norm=True, bias=False)

    def forward(self, voints_feats,voints_mask=None, view_info=None,xyz=None):
        bs,_,nb_voints,nb_views = voints_feats.shape
        # if not voints_mask:
        #     voints_mask = torch.ones((bs, nb_voints, nb_views)).cuda()
        if self.use_xyz:
            voints_feats = torch.cat((xyz[..., None].repeat(1, 1, 1, nb_views), voints_feats), dim=1)
        if self.view_embed.use_view_info:
            voints_feats = torch.cat((self.view_embed(view_info).transpose(
                1, 2).unsqueeze(2).repeat(1, 1, nb_voints,1), voints_feats), dim=1)
        voints_feats = self.feature_extractor(voints_feats, voints_mask)
        if self.use_cls_voint:
            cls_voints_feats = self.aux_classifier(torch.max(voints_feats, dim=2)[0].unsqueeze(2), voints_mask[:,0, :].unsqueeze(1))
            voints_feats = torch.cat((cls_voints_feats.repeat(1, 1, nb_voints,1), voints_feats), dim=1)
        points_feats = self.aggr(self.classifier(voints_feats, voints_mask), voints_mask)
        return points_feats


class VointGCN(nn.Module):
    def __init__(self, in_size, out_size, feat_dim=64, viewembedder=None, aggr="max", use_cls_voint=False, leanred_cls_token=False, use_xyz=False,voint_depth=2):
        super().__init__()
        self.use_cls_voint = use_cls_voint
        self.use_xyz = use_xyz
        if use_xyz:
            in_size += 3 
        self.view_embed = viewembedder
        in_size += self.view_embed.added_dim()
        self.input_embedder = VointConv([in_size,feat_dim], act='relu', norm=True, bias=True, drop=0.0, )
        self.feature_extractor = VointGraphConv(
            [feat_dim]*int(voint_depth+1), act='relu', norm=True, bias=True, drop=0.0, aggr=aggr)
        if use_cls_voint:
            self.aux_classifier = VointConv(
                [feat_dim, feat_dim, feat_dim], act='relu', norm=True, bias=True)
            self.classifier = VointConv(
                [feat_dim*2, feat_dim, out_size], act='relu', norm=False, bias=False)
        else:
            self.aux_classifier = nn.Sequential()
            self.classifier = VointConv(
                [feat_dim, out_size], act='relu', norm=False, bias=False,)
        if leanred_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, feat_dim, 1, 1)).cuda()
        else:
            self.cls_token = torch.zeros((1, feat_dim, 1, 1)).cuda()

    def forward(self, voints_feats, voints_mask=None, view_info=None, xyz=None):
        bs, C , nb_voints, nb_views = voints_feats.shape
        # if not voints_mask:
        #     voints_mask = torch.ones((bs, nb_voints, nb_views)).cuda()
        if self.use_xyz:
            voints_feats = torch.cat((xyz[..., None].repeat(1, 1, 1, nb_views), voints_feats), dim=1)
        if self.view_embed.use_view_info:
            voints_feats = torch.cat((self.view_embed(view_info).transpose(
                1, 2).unsqueeze(2).repeat(1, 1, nb_voints, 1), voints_feats), dim=1)
        voints_feats = self.input_embedder(voints_feats, voints_mask)
        voints_feats = torch.cat((self.cls_token.repeat(bs,1,nb_voints,1), voints_feats), dim=-1)
        voints_mask = torch.cat((torch.ones((bs, nb_voints, 1)).cuda(), voints_mask), dim=-1)
        voints_feats = self.feature_extractor(voints_feats, voints_mask)
        if self.use_cls_voint:
            cls_voints_feats = self.aux_classifier(torch.max(voints_feats, dim=2)[0].unsqueeze(2), voints_mask[:, 0, :].unsqueeze(1))
            voints_feats = torch.cat((cls_voints_feats.repeat(1, 1, nb_voints, 1), voints_feats), dim=1)
        points_feats = self.classifier(voints_feats, voints_mask)[:,:,:,0]

        return points_feats


class VointGAT(nn.Module):
    def __init__(self, in_size, out_size, feat_dim=64, viewembedder=None, aggr="max", use_cls_voint=False, leanred_cls_token=False, use_xyz=False, voint_depth=2):
        super().__init__()
        self.use_cls_voint = use_cls_voint
        self.use_xyz = use_xyz
        if use_xyz:
            in_size += 3 
        self.view_embed = viewembedder
        in_size += self.view_embed.added_dim()
        self.input_embedder = VointConv(
            [in_size, feat_dim], act='relu', norm=True, bias=True, drop=0.0, )
        self.feature_extractor = VointGraphAttention(
            [feat_dim]*int(voint_depth+1), act='relu', norm=True, bias=True, drop=0.0, aggr=aggr)
        if use_cls_voint:
            self.aux_classifier = VointConv(
                [feat_dim, feat_dim, feat_dim], act='relu', norm=True, bias=True)
            self.classifier = VointConv(
                [feat_dim*2, feat_dim, out_size], act='relu', norm=False, bias=False)
        else:
            self.aux_classifier = nn.Sequential()
            self.classifier = VointConv(
                [feat_dim, out_size], act='relu', norm=False, bias=False,)
        if leanred_cls_token:
            self.cls_token = nn.Parameter(
                torch.randn(1, feat_dim, 1, 1)).cuda()
        else:
            self.cls_token = torch.zeros((1, feat_dim, 1, 1)).cuda()

    def forward(self, voints_feats, voints_mask=None, view_info=None, xyz=None):
        bs, C, nb_voints, nb_views = voints_feats.shape
        # if not voints_mask:
        #     voints_mask = torch.ones((bs, nb_voints, nb_views)).cuda()
        if self.use_xyz:
            voints_feats = torch.cat((xyz[..., None].repeat(1, 1, 1, nb_views), voints_feats), dim=1)
        if self.view_embed.use_view_info:
            voints_feats = torch.cat((self.view_embed(view_info).transpose(
                1, 2).unsqueeze(2).repeat(1, 1, nb_voints, 1), voints_feats), dim=1)
        voints_feats = self.input_embedder(voints_feats, voints_mask)
        voints_feats = torch.cat((self.cls_token.repeat(
            bs, 1, nb_voints, 1), voints_feats), dim=-1)
        voints_mask = torch.cat(
            (torch.ones((bs, nb_voints, 1)).cuda(), voints_mask), dim=-1)
        voints_feats = self.feature_extractor(voints_feats, voints_mask)
        if self.use_cls_voint:
            cls_voints_feats = self.aux_classifier(torch.max(voints_feats, dim=2)[
                                                   0].unsqueeze(2), voints_mask[:, 0, :].unsqueeze(1))
            voints_feats = torch.cat((cls_voints_feats.repeat(
                1, 1, nb_voints, 1), voints_feats), dim=1)
        points_feats = self.classifier(voints_feats, voints_mask)[:, :, :, 0]

        return points_feats

class VointFormer(nn.Module):
    def __init__(self, in_size, out_size, feat_dim=64, viewembedder=None, aggr="max", use_cls_voint=False, leanred_cls_token=False, use_xyz=False, voint_depth=2):
        super().__init__()
        self.use_cls_voint = use_cls_voint
        self.use_xyz = use_xyz
        if use_xyz:
            in_size += 3 
        self.view_embed = viewembedder
        in_size += self.view_embed.added_dim()
        self.input_embedder = VointConv(
            [in_size, feat_dim], act='relu', norm=True, bias=True, drop=0.0, )
        self.feature_extractor = VointSelfAttention(
            [feat_dim]*int(voint_depth+1), act='relu', norm=True, bias=True, drop=0.0, aggr=aggr)
        if use_cls_voint:
            self.aux_classifier = VointConv(
                [feat_dim, feat_dim, feat_dim], act='relu', norm=True, bias=True)
            self.classifier = VointConv(
                [feat_dim*2, feat_dim, out_size], act='relu', norm=False, bias=False)
        else:
            self.aux_classifier = nn.Sequential()
            self.classifier = VointConv(
                [feat_dim, out_size], act='relu', norm=False, bias=False,)
        if leanred_cls_token:
            self.cls_token = nn.Parameter(
                torch.randn(1, feat_dim, 1, 1)).cuda()
        else:
            self.cls_token = torch.zeros((1, feat_dim, 1, 1)).cuda()

    def forward(self, voints_feats, voints_mask, view_info=None, xyz=None):
        bs, C, nb_voints, nb_views = voints_feats.shape
        if self.use_xyz:
            voints_feats = torch.cat((xyz[..., None].repeat(1, 1, 1, nb_views), voints_feats), dim=1)
        if self.view_embed.use_view_info:
            voints_feats = torch.cat((self.view_embed(view_info).transpose(
                1, 2).unsqueeze(2).repeat(1, 1, nb_voints, 1), voints_feats), dim=1)
        voints_feats = self.input_embedder(voints_feats, voints_mask)
        voints_feats = torch.cat((self.cls_token.repeat(
            bs, 1, nb_voints, 1), voints_feats), dim=-1)
        voints_mask = torch.cat(
            (torch.ones((bs, nb_voints, 1)).cuda(), voints_mask), dim=-1)
        voints_feats = self.feature_extractor(voints_feats, voints_mask)
        if self.use_cls_voint:
            cls_voints_feats = self.aux_classifier(torch.max(voints_feats, dim=2)[
                                                   0].unsqueeze(2), voints_mask[:, 0, :].unsqueeze(1))
            voints_feats = torch.cat((cls_voints_feats.repeat(
                1, 1, nb_voints, 1), voints_feats), dim=1)
        points_feats = self.classifier(voints_feats, voints_mask)[:, :, :, 0]

        return points_feats