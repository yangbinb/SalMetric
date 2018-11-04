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
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
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
        print 'conv1', conv_1.shape
        conv_2 = self.conv_2(conv_1)
        print 'conv2', conv_2.shape
        conv_3 = self.conv_3(conv_2)
        print 'conv3', conv_3.shape
        conv_4 = self.conv_4(conv_3)
        print 'conv4', conv_4.shape
        conv_5 = self.conv_5(conv_4)
        print 'conv5', conv_5.shape

        global_ = self.global_(conv_5)
        print 'global', global_.shape
        global_score = self.global_score(global_)
        print 'global score', global_score.shape

        conv_6 = self.conv_6(conv_1)
        print 'conv_6', conv_6.shape
        conv_7 = self.conv_7(conv_2)
        print 'conv_7', conv_7.shape
        conv_8 = self.conv_8(conv_3)
        print 'conv_8', conv_8.shape
        conv_9 = self.conv_9(conv_4)
        print 'conv_9', conv_9.shape
        conv_10 = self.conv_10(conv_5)
        print 'conv_10', conv_10.shape

        contrast_1 = conv_6 - self.contrast_pool_1(conv_6)
        print 'contrast1', contrast_1.shape
        contrast_2 = conv_7 - self.contrast_pool_2(conv_7)
        print 'contrast2', contrast_2.shape
        contrast_3 = conv_8 - self.contrast_pool_3(conv_8)
        print 'contrast3', contrast_3.shape
        contrast_4 = conv_9 - self.contrast_pool_4(conv_9)
        print 'contrast4', contrast_4.shape
        contrast_5 = conv_10 - self.contrast_pool_5(conv_10)
        print 'contrast5', contrast_5.shape

        unpool_5 = self.unpooling_5(torch.cat((conv_10, contrast_5), dim=1))
        print 'unpool5', unpool_5.shape
        unpool_4 = self.unpooling_4(torch.cat((conv_9, contrast_4, unpool_5), dim=1))
        print 'unpool4', unpool_4.shape
        unpool_3 = self.unpooling_3(torch.cat((conv_8, contrast_3, unpool_4), dim=1))
        print 'unpool3', unpool_3.shape
        unpool_2 = self.unpooling_2(torch.cat((conv_7, contrast_2, unpool_3), dim=1))
        print 'unpool2', unpool_2.shape

        local = self.local(torch.cat((conv_6, contrast_1, unpool_2), dim=1))
        print 'local', local.shape
        local_score = self.local_score(local)
        print 'local_score', local_score.shape
        score = global_score + local_score
        score = F.upsample(score, scale_factor=2, mode='bilinear', align_corners=True)
        return score