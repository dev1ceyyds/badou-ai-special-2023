import torch
import torch.nn as nn
from dcmtracking.detection.yolov5.nets.CSPdarknet import C3, Conv, CSPDarknet


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict          = {'n': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'n': 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)
        base_depth          = max(round(dep_mul * 3), 1)

        self.backbone       = CSPDarknet(base_channels, base_depth, phi, pretrained)
            
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)


        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)

        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)

        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

    def forward(self, x):

        feat1, feat2, feat3 = self.backbone(x)

        P5 = self.conv_for_feat3(feat3)

        P5_upsample = self.upsample(P5)

        P4 = torch.cat([P5_upsample, feat2], 1)

        P4 = self.conv3_for_upsample1(P4)

        P4 = self.conv_for_feat2(P4)

        P4_upsample = self.upsample(P4)

        P3 = torch.cat([P4_upsample, feat1], 1)

        P3 = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)

        P4 = torch.cat([P3_downsample, P4], 1)

        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)

        P5 = torch.cat([P4_downsample, P5], 1)

        P5 = self.conv3_for_downsample2(P5)

        out2 = self.yolo_head_P3(P3)

        out1 = self.yolo_head_P4(P4)

        out0 = self.yolo_head_P5(P5)
        return out0, out1, out2

