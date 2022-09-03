import torch
from mvtorch.mvaggregate import MVAggregate
from mvtorch.models.voint import PointMLPClassifier

class MVNetwork(torch.nn.Module):

    def __init__(self, num_classes, mode='cls', pretraining=True, net_name='resnet18', agr_type='max', lifting_net=torch.nn.Sequential()):
        super().__init__()

        self.num_classes = num_classes
        self.mode = mode
        self.pretraining = pretraining
        self.net_name = net_name
        self.agr_type = agr_type
        self.lifting_net = lifting_net
        
        if self.mode == 'cls':
            if self.net_name.startswith('resnet'):
                depth_to_feat_dim = {'18': 512, '34': 512, '50': 2048, '101': 2048, '152': 2048}
                assert self.net_name[6:] in list(depth_to_feat_dim.keys()), 'The requested resnet depth is not available'
                self.feat_dim = depth_to_feat_dim[self.net_name[6:]]
                
                network = torch.hub.load('pytorch/vision:v0.10.0', self.net_name, pretrained=self.pretraining)
                network.fc = torch.nn.Sequential()

                self.mvnetwork = MVAggregate(
                    model=network,
                    agr_type=self.agr_type, 
                    feat_dim=self.feat_dim, 
                    num_classes=self.num_classes, 
                    lifting_net=self.lifting_net
                )
            else:
                raise ValueError('Invalid classification network name')
        else:
            raise ValueError('Invalid mode for mutli-view network')

    def forward(self, x):
        return self.mvnetwork(x)

class MLPClassifier(torch.nn.Module):

    def __init__(self, num_classes, num_parts, in_size=64, use_xyz=False, use_global=False, skip=True, parallel_head=True, feat_dim=64):
        super().__init__()

        self.num_classes = num_classes
        self.in_size = in_size
        self.out_size = num_parts + 1
        self.use_xyz = use_xyz
        self.use_global = use_global
        self.skip = skip
        self.parallel_head = parallel_head
        self.feat_dim = feat_dim

        self.mlp_classifier = PointMLPClassifier(
            self.in_size, 
            self.out_size, 
            use_xyz=self.use_xyz,
            use_global=self.use_global, 
            skip=self.skip, 
            nb_heads=self.num_classes, 
            parallel_head=self.parallel_head, 
            feat_dim=self.feat_dim, 
            extra_net=torch.nn.Sequential(),
        )

    def forward(self, points_feats, xyz=None, cls=None):
        return self.mlp_classifier(points_feats, xyz=xyz, cls=cls)
