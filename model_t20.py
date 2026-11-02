import segmentation_models_pytorch as smp


def build_model():

    model = smp.Unet(
        encoder_name="resnet34",

        # 和10phase实验一样
        encoder_weights=None,

        # 唯一核心变化
        in_channels=1,

        # 输出仍然是一张FDK
        classes=1,

        activation=None,
    )

    return model