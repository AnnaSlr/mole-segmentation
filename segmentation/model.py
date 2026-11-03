"""U-Net with a pretrained encoder."""
import segmentation_models_pytorch as smp


def build_model(encoder: str = "resnet34", pretrained: bool = True):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
    )