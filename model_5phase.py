import segmentation_models_pytorch as smp


def build_model():

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,

        # T00,T20,T40,T60,T80
        in_channels=5,

        classes=1,
        activation=None,
    )

    return model