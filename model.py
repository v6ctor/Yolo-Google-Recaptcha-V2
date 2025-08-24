import torch
import torch.nn as nn 
import numpy as np
from torch.cuda.amp import autocast

from utils import intersection_over_union

"""
Code is inspired by findings from https://arxiv.org/pdf/1506.02640 and YOLOv3 paper

Datasets from 
    1. (500 YOLO labeled reCaptcha v2) - https://www.kaggle.com/datasets/mikhailma/test-dataset
    2. (unlabeled reCaptcha v2) - https://www.kaggle.com/datasets/cry2003/google-recaptcha-v2-images
"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_threshold, is_batch_norm=True, **kwargs):
        super(ConvBlock, self).__init__()

        self.use_batch_norm = is_batch_norm

        if is_batch_norm:
            self.conv_layer = nn.Conv2d(in_channels, out_channels, bias= not is_batch_norm, **kwargs)
            self.batch_norm = nn.BatchNorm2d(out_channels)
        
        else:
            self.layer = nn.Conv2d(in_channels, out_channels, **kwargs)

        if dropout_threshold > 0:
            self.dropout = nn.Dropout(dropout_threshold)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        if self.use_batch_norm:
            x = self.leaky_relu(self.batch_norm(self.conv_layer(x)))
        else:
            x = self.leaky_relu(self.layer(x))

        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, channels, is_residual=True, num_repeats=1):
        super(ResidualBlock, self).__init__()

        self.layers = nn.ModuleList()
        self.use_residual = is_residual
        self.num_repeats = num_repeats

        for _ in range(num_repeats):
            self.layers.append(
                nn.Sequential(
                    ConvBlock(channels, channels // 2, kernel_size=1, dropout_threshold=0.2),
                    ConvBlock(channels // 2, channels, kernel_size=3, padding=1, dropout_threshold=0.2)
                )
            )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)

        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ScalePrediction, self).__init__()
        self.num_classes= num_classes
        self.prediction = nn.Sequential(
            ConvBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1, dropout_threshold=0.2),
            ConvBlock(2 * in_channels, (num_classes + 5) * 3, kernel_size=1, is_batch_norm=False, dropout_threshold=0.2)
        )

    def forward(self, x):
        return (
            self.prediction(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

class YOLO(nn.Module):
    def __init__(self, conv_params, in_channels=3, num_classes=3):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.layers = nn.ModuleList()

        for param in conv_params:
            print(f"Building: ", param)
            if isinstance(param, tuple):
                out_channels, kernel_size, stride = param 
                self.layers.append(ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dropout_threshold=0.1, padding=1 if kernel_size == 3 else 0, stride=stride))
                in_channels = out_channels
            elif isinstance(param, list):
                self.layers.append(ResidualBlock(in_channels, num_repeats=param[1]))
            elif isinstance(param, str):
                if param == "scaled_prediction":
                    self.layers.append(ResidualBlock(in_channels, is_residual=False, num_repeats=1))
                    self.layers.append(ConvBlock(in_channels, in_channels // 2, kernel_size=1, dropout_threshold=0.2)),
                    self.layers.append(ScalePrediction(in_channels // 2, num_classes=self.num_classes))

                    in_channels = in_channels // 2

                elif param == "upsample":
                    self.layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3
        
    def forward(self, x):
        with torch.amp.autocast("cuda"):
            outputs = []
            route_connections = []

            for layer in self.layers:
                if isinstance(layer, ScalePrediction):
                    outputs.append(layer(x))
                    continue

                x = layer(x)

                if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                    route_connections.append(x)

                elif isinstance(layer, nn.Upsample):
                    x = torch.cat([x, route_connections[-1]], dim=1)
                    route_connections.pop()

        return outputs

# The `Loss` class in Python defines a custom loss function for a neural network model that includes
# components for object detection and classification tasks.
class Loss(nn.Module):
    def __init__(self, lambda_class=5, lambda_obj=5, lambda_noobj=1, lambda_box=5):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = lambda_class
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_box = lambda_box


    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0
        
        no_object_loss = self.bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]))

        # Object loss:
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_predictions = torch.cat([self.sigmoid(predictions[...,1:3]), torch.exp(predictions[..., 3:5] * anchors)], dim=-1)
        ious = intersection_over_union(box_predictions[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]))

        # Box coordinate loss
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # Class loss
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        return (
            self.lambda_box * box_loss + self.lambda_obj * object_loss + self.lambda_noobj * no_object_loss + self.lambda_class * class_loss
        )
    
if __name__ == "__main__":
    # Config is a list of tuples: (out_channels, kernel_size, stride) corresponding to the 
    # Darknet-53 architecture
    darknet53_config = [
        (32, 3, 1),
        (64, 3, 2),
        ["residual_block", 1],
        (128, 3, 2),
        ["residual_block", 2],
        (256, 3, 2),
        ["residual_block", 8],
        (512, 3, 2),
        ["residual_block", 8],
        (1024, 3, 2),
        ["residual_block", 4],
    ]

    conv = [
        (512, 1, 1),
        (1024, 3, 1),
        "scaled_prediction",
        (256, 1, 1),
        "upsample",
        (256, 1, 1),
        (512, 3, 1),
        "scaled_prediction",
        (128, 1, 1),
        "upsample",
        (128, 1, 1),
        (256, 3, 1),
        "scaled_prediction",
    ]

    image_size = 120

    model = YOLO(darknet53_config + conv, num_classes=20)

    x = torch.randn((2, 3, image_size, image_size))
    out = model(x)