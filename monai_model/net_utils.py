import custom_unet
import local_unet
import monai.losses


def load_net(args, include_loss=False):
    # create net, loss, optimizer
    if args.model == 'unet':
        net = custom_unet.UNet(
                    dimensions=3,
                    in_channels=4,
                    out_channels=3,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                    up_mode=args.up_mode,
                )
        loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True)
    elif args.model == 'local_unet':
        net = local_unet.UNet(
                    dimensions=3,
                    in_channels=4,
                    out_channels=3,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                    up_mode=args.up_mode,
                )
        loss_fn = monai.losses.MaskedDiceLoss(sigmoid=True, squared_pred=True)
    if include_loss:
        return net, loss_fn
    else:
        return net
