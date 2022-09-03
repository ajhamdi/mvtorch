import torch
from mvtorch import mvaggregate

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

                self.mvnetwork = mvaggregate.MVAggregate(
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