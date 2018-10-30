import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class NLDF(nn.Module):
    def __init__(self):
        super(NLDF, self).__init__()
        # 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_6 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_9 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.conv_10 = nn.Conv2d(512, 128, kernel_size=3, padding=1)

        self.contrast_pool_1 = nn.AvgPool2d(3, stride=1, padding=1)
        self.contrast_pool_2 = nn.AvgPool2d(3, stride=1, padding=1)
        self.contrast_pool_3 = nn.AvgPool2d(3, stride=1, padding=1)
        self.contrast_pool_4 = nn.AvgPool2d(3, stride=1, padding=1)
        self.contrast_pool_5 = nn.AvgPool2d(3, stride=1, padding=1)

        self.unpooling_5 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.unpooling_4 = nn.Sequential(
            nn.ConvTranspose2d(128 * 2 + 128, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.unpooling_3 = nn.Sequential(
            nn.ConvTranspose2d(128 * 2 + 256, 384, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.unpooling_2 = nn.Sequential(
            nn.ConvTranspose2d(128 * 2 + 384, 512, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True)
        )

        self.local = nn.Conv2d(128 * 2 + 512, 640, kernel_size=1, stride=1, padding=0)
        self.local_score = nn.Conv2d(640, 2, kernel_size=1, stride=1, padding=0)

        self.global_ = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=5, stride=1, padding=0),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        )
        self.global_score = nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        conv_3 = self.conv_3(conv_2)
        conv_4 = self.conv_4(conv_3)
        conv_5 = self.conv_5(conv_4)

        global_ = self.global_(conv_5)
        global_score = self.global_score(global_)

        conv_6 = self.conv_6(conv_1)
        conv_7 = self.conv_7(conv_2)
        conv_8 = self.conv_8(conv_3)
        conv_9 = self.conv_9(conv_4)
        conv_10 = self.conv_10(conv_5)

        contrast_1 = self.contrast_pool_1(conv_6)
        contrast_2 = self.contrast_pool_2(conv_7)
        contrast_3 = self.contrast_pool_2(conv_8)
        contrast_4 = self.contrast_pool_2(conv_9)
        contrast_5 = self.contrast_pool_2(conv_10)

        unpool_5 = self.unpooling_5(torch.cat((conv_10, contrast_5), dim=1))
        unpool_4 = self.unpooling_4(torch.cat((conv_9, contrast_4, unpool_5), dim=1))
        unpool_3 = self.unpooling_3(torch.cat((conv_8, contrast_3, unpool_4), dim=1))
        unpool_2 = self.unpooling_2(torch.cat((conv_7, contrast_2, unpool_3), dim=1))

        local = self.local(torch.cat((conv_6, contrast_1, unpool_2), dim=1))
        local_score = self.local_score(local)
        score = global_score + local_score
        score = F.upsample(score, scale_factor=2, mode='bilinear', align_corners=True)
        return score