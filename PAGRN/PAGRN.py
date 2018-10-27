import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def Batch_ReLU_Pooling(n_channels):
    return nn.Sequential(
        nn.BatchNorm2d(n_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class PAGRN(nn.Module):
    def __init__(self):
        super(PAGRN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        self.attention_5_1 = nn.Conv2d(512, 512, kernel_size=1)
        self.attention_5_2 = nn.Conv2d(512, 512, kernel_size=1)
        self.attention_5_3 = nn.Conv2d(512, 256, kernel_size=1)

        self.attention_4_1 = nn.Conv2d(256, 256, kernel_size=1)
        self.attention_4_2 = nn.Conv2d(256, 256, kernel_size=1)
        self.attention_4_3 = nn.Conv2d(256, 256, kernel_size=1)

        self.attention_3_1 = nn.Conv2d(256, 256, kernel_size=1)
        self.attention_3_2 = nn.Conv2d(256, 256, kernel_size=1)
        self.attention_3_3 = nn.Conv2d(256, 256, kernel_size=1)

        self.saliency_map = nn.Conv2d(256, 1, kernel_size=1)

        self.recurrent_5 = nn.Conv2d(512, 512, kernel_size=1)
        self.recurrent_4 = nn.Conv2d(512, 512, kernel_size=1)
        self.recurrent_3 = nn.Conv2d(512, 256, kernel_size=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        conv3_4 = self.block3(x)
        x = Batch_ReLU_Pooling(256)(conv3_4)
        conv4_4 = self.block4(x)
        x = Batch_ReLU_Pooling(512)(conv4_4)
        conv5_5 = self.block5(x)
        conv5_5 = nn.BatchNorm2d(512)(conv5_5)

        gap_5 = F.adaptive_avg_pool2d(conv5_5, (1,1))
        coeff = self.attention_5_1(gap_5)
        fca_5 = conv5_5*coeff
        fca_5_att_mask = self.attention_5_2(fca_5)
        fcsa_5 = fca_5*fca_5_att_mask
        fcsa_5_up = F.upsample(fcsa_5, scale_factor=2, mode='bilinear', align_corners=True)
        add_5 = self.attention_5_3(fcsa_5_up + conv4_4)

        gap_4 = F.adaptive_avg_pool2d(add_5, (1,1))
        coeff = self.attention_4_1(gap_4)
        fca_4 = add_5 * coeff
        fca_4_att_mask = self.attention_4_2(fca_4)
        fcsa_4 = fca_4 * fca_4_att_mask
        fcsa_4_up = F.upsample(fcsa_4, scale_factor=2, mode='bilinear', align_corners=True)
        add_4 = self.attention_4_3(fcsa_4_up + conv3_4)

        gap_3 = F.adaptive_avg_pool2d(add_4, (1, 1))
        coeff = self.attention_3_1(gap_3)
        fca_3 = add_4 * coeff
        fca_3_att_mask = self.attention_3_2(fca_3)
        fcsa_3 = fca_3 * fca_3_att_mask

        salient_map = self.saliency_map(fcsa_3)
        salient_map = F.upsample(salient_map, scale_factor=4, mode='bilinear', align_corners=True)
        print salient_map.shape

        return salient_map
