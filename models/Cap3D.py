import torch
import torch.nn as nn
from models.capsule_layer import CapsuleLayer
import models.nn_

class Cap3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2, bias=False)
        )
        # self.PrimaryCap = CapsuleLayer(1, 16, "conv", k=5, s=1, t_1=2, z_1=16, routing=1)

        self.DownStep_1 = nn.Sequential( # 1/2
            CapsuleLayer(1, 16, "conv", k=5, s=2, t_1=2, z_1=16, routing=1),
            CapsuleLayer(2, 16, "conv", k=5, s=1, t_1=4, z_1=16, routing=3),
        )

        self.DownStep_2 = nn.Sequential( # 1/4
            CapsuleLayer(4, 16, "conv", k=5, s=2, t_1=4, z_1=32, routing=3),
            CapsuleLayer(4, 32, "conv", k=5, s=1, t_1=8, z_1=32, routing=3),
        )
        self.DownStep_3 = nn.Sequential(  # 1/8
            CapsuleLayer(8, 32, "conv", k=5, s=2, t_1=8, z_1=64, routing=3),
            CapsuleLayer(8, 64, "conv", k=5, s=1, t_1=8, z_1=32, routing=3)
        )
        self.UpConv_1 = CapsuleLayer(8, 32, "deconv", k=5, s=2, t_1=8, z_1=32, routing=3)
        self.TransConv_1 = CapsuleLayer(16, 32, "conv", k=5, s=1, t_1=4, z_1=32, routing=3)

        self.UpConv_2 = CapsuleLayer(4, 32, "deconv", k=5, s=2, t_1=4, z_1=16, routing=3)
        self.TransConv_2 = CapsuleLayer(8, 16, "conv", k=5, s=1, t_1=4, z_1=16, routing=3)

        self.UpConv_3 = CapsuleLayer(4, 16, "deconv", k=5, s=2, t_1=2, z_1=16, routing=3)

        self.TransConv_3 = CapsuleLayer(3, 16, "conv", k=5, s=1, t_1=2, z_1=16, routing=3)


    def forward(self,x):
        x = self.conv_1(x)
        x.unsqueeze_(1)

        skip_1 = x #[N,1,16,H,W]

        x = self.DownStep_1(x)
        skip_2 = x #[N,4,16,H/2,W/2]

        x = self.DownStep_2(x)
        skip_3 = x  # [N,8,32,H/4,W/4]

        x = self.DownStep_3(x) # [N,8,32,H/8,W/8]

        x = self.UpConv_1(x) # [N,8,32,H/4,W/4]
        x = torch.cat((x,skip_3),1) # [N,16,32,H/4,W/4]
        x = self.TransConv_1(x) #[N,4,32,H/4,W/4]

        x = self.UpConv_2(x) #[N,4,16,H/2,W/2]
        x = torch.cat((x,skip_2),1) #[N,8,16,H/2,W/2]
        x = self.TransConv_2(x) #[N,4,16,H/2,W/2]

        x = self.UpConv_3(x) # [N,2,16,H,W]
        x = torch.cat((x,skip_1),1) # [N,3,16,H,W]
        x = self.TransConv_3(x) # [N,2,16,H,W]
        print(x.shape)

        feature_map_1 = x[:, 0, :, :, :]
        feature_map_2 = x[:, 1, :, :, :]
        feature_map_1 = self.compute_vector_length(feature_map_1).squeeze(1)
        feature_map_2 = self.compute_vector_length(feature_map_2).squeeze(1)
        return (feature_map_1,feature_map_2)

    def compute_vector_length(self, x):
        out = (x.pow(2)).sum(1, True)+1e-9
        out=out.sqrt()
        return out



def test():
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    model = Cap3D()
    # model = model.cuda()
    print(model)
    # c = input('s')
    a = torch.ones(1, 3, 256, 256)
    # a = a.cuda()
    b1,b2 = model(a)
    print('b1',b1)
    print('b2',b2)
    c1 = b1.sum()
    print('c1',c1)
    c1.backward()
    for k,v in model.named_parameters():
        print(v.grad,k)
        break
    # from tensorboardX import SummaryWriter
    # with SummaryWriter(comment='LeNet') as w:
    #     w.add_graph(model, a)
    # print(b1.shape)
    # print(b1)
# test()


def compute_vector_length( x):
    out = (x.pow(2)).sum(1, True)+1e-9
    out.sqrt_()
    return out