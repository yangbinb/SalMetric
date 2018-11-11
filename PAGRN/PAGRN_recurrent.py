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
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
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

        self.recurrent_4 = nn.Conv2d(512, 512, kernel_size=1)
        self.recurrent_3 = nn.Conv2d(512, 256, kernel_size=1)
        self.recurrent_2 = nn.Conv2d(512, 128, kernel_size=1)

        self.batch_normal_pooling_2 = Batch_ReLU_Pooling(128)
        self.batch_normal_2 = nn.BatchNorm2d(128)
        self.batch_normal_pooling_3 = Batch_ReLU_Pooling(256)
        self.batch_normal_3 = nn.BatchNorm2d(256)
        self.batch_normal_pooling_4 = Batch_ReLU_Pooling(512)
        self.batch_normal_4 = nn.BatchNorm2d(512)

    def forward(self, x, H):
        x = self.block1(x)
        conv2_2 = self.block2(x)
        x = self.batch_normal_pooling_2(conv2_2)
        x += self.batch_normal_2(self.recurrent_2(F.upsample(H, scale_factor=4, mode='bilinear', align_corners=True)))
        conv3_4 = self.block3(x)
        x = self.batch_normal_pooling_3(conv3_4)
        x += self.batch_normal_3(self.recurrent_3(F.upsample(H, scale_factor=2, mode='bilinear', align_corners=True)))
        conv4_4 = self.block4(x)
        x = self.batch_normal_pooling_4(conv4_4)
        x += self.batch_normal_4(self.recurrent_4(H))
        conv5_5 = self.block5(x)
        new_H = conv5_5


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

        return salient_map, new_H
