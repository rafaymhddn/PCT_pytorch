import torch
import torch.nn as nn
import torch.nn.functional as F

from module import Embedding, NeighborEmbedding, OA, SA


class NaivePCT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embedding(3, 128)

        self.sa1 = SA(128)
        self.sa2 = SA(128)
        self.sa3 = SA(128)
        self.sa4 = SA(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class Segmentation(nn.Module):
    def __init__(self, part_num):
        super().__init__()

        self.part_num = part_num

        self.label_conv = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)

        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(0.5)
    
    def forward(self, x, x_max, x_mean):
        batch_size, _, N = x.size()

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)

        #cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        #cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        x = torch.cat([x, x_max_feature, x_mean_feature,], dim=1)  # 1024 * 3 + 64

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        return x



"""
Part Segmentation Networks.
"""

class NaivePCTSeg(nn.Module):
    def __init__(self, part_num=50):
        super().__init__()
    
        self.encoder = NaivePCT()
        self.seg = Segmentation(part_num)

    def forward(self, x):
        x, x_max, x_mean = self.encoder(x)
        x = self.seg(x, x_max, x_mean)
        return x



if __name__ == '__main__':
    pc = torch.rand(4, 3, 1024).to('cuda')
    


    # testing for segmentation networks
    naive_pct_seg = NaivePCTSeg().to('cuda')

    print(naive_pct_seg(pc).size())
    print(pc.size())


    # testing for normal estimation networks
