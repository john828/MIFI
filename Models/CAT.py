import torch.nn as nn
import torch


class CAT(nn.Module):
    def __init__(self):
        super(CAT, self).__init__()

        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(0.5)

        self.logits = nn.Conv3d(in_channels=(384 + 384 + 128 + 128), out_channels=16,kernel_size=[1, 1, 1],padding=0,bias=True,)
        # self.logits = nn.Conv3d(in_channels=(384 + 384 + 128 + 128), out_channels=16,kernel_size=[1, 1, 1],padding=0,bias=True,)
    def forward(self, x1,x2):

        x = torch.cat((x1,x2),dim=2)
        # x = torch.cat((x1, x2), dim=1)
        # x = x1 + x2
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.logits(x)
        x = x.squeeze(3).squeeze(3).mean(2)

        return x

