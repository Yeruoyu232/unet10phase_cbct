import segmentation_models_pytorch as smp


def build_model():

    model = smp.Unet(

        # 标准U-Net + ResNet34 encoder
        encoder_name="resnet34",

        # 不使用ImageNet预训练
        encoder_weights=None,

        # 10个呼吸phase
        in_channels=10,

        # 输出1张灰度FDK
        classes=1,

        # 图像回归，不加sigmoid
        activation=None,
    )

    return model