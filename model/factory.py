import torch.nn as nn
import torchvision.models.segmentation as models

class ModelFactory:
    """
    Factory class to create segmentation models.

    Supported models:
    - deeplabv3_resnet50
    - deeplabv3_resnet101
    - maskrcnn_resnet50

    Usage:
        model = ModelFactory.create_model(name="deeplabv3_resnet50", num_classes=6, pretrained_weights="COCO")
    """

    @staticmethod
    def create_model(name: str, num_classes: int, pretrained_weights: str = None) -> nn.Module:
        """
        Create and return a segmentation model.

        Args:
            name (str): Model name ("deeplabv3_resnet50", "deeplabv3_resnet101", "maskrcnn_resnet50").
            num_classes (int): Number of output classes.
            pretrained_weights (str, optional): "COCO", "Cityscapes", or None.

        Returns:
            torch.nn.Module: Initialized model.
        """
        name = name.lower()
        pretrained = False

        if pretrained_weights is not None:
            pretrained = True

        if name == "deeplabv3_resnet50":
            model = models.deeplabv3_resnet50(weights="DEFAULT" if pretrained else None)
            model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

        elif name == "deeplabv3_resnet101":
            model = models.deeplabv3_resnet101(weights="DEFAULT" if pretrained else None)
            model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

        elif name == "maskrcnn_resnet50":
            model = models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
            # MaskRCNN expects a dict with bounding boxes & masks
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
                in_features_mask, hidden_layer, num_classes
            )

        else:
            raise ValueError(f"Unsupported model name: {name}")

        return model

class DeepLabV3Wrapper(nn.Module):
    """
    Wrapper around DeepLabV3 models to return only the 'out' tensor for loss computation.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['out']
