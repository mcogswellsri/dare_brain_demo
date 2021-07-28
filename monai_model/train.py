# based on https://github.com/Project-MONAI/MONAI/blob/master/examples/segmentation_3d_ignite/unet_training_dict.py
import os
import sys
import logging
import numpy as np
import torch
import ignite
import ignite.metrics
import monai

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping

from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    TensorBoardImageHandler,
    MeanDice,
    stopping_fn_from_metric,
)
from monai.data import DataLoader, create_test_image_3d, list_data_collate
from monai.networks import predict_segmentation

import options
import local_unet
import custom_unet

from data_utils import load_data
from net_utils import load_net


def prepare_batch(batch, device, args):
    batch = dict(batch)
    for k, tensor in batch.items():
        if torch.is_tensor(tensor):
            batch[k] = tensor.to(device)
    if args.model == 'unet':
        x = batch['image']
    elif args.model == 'local_unet':
        x = (batch["image"], batch["mask"])
    if "seg" in batch:
        seg = batch["seg"]
    else:
        seg = None
    return x, seg, batch


def prepare_output(batch, y, y_pred, loss):
    output = dict(batch)
    output["y"] = batch["seg"] if "seg" in batch else None
    output["y_pred"] = y_pred
    output["loss"] = loss
    return output


def main(args):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    monai.utils.set_determinism(seed=0)
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(args.model_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # load data
    train_ds, train_loader, val_ds, val_loader = load_data(args)

    # create net, loss, optimizer
    net, loss_fn = load_net(args, include_loss=True)
    device = torch.device("cuda:0")
    net = net.to(device)
    lr = 1e-4
    optimizer= torch.optim.Adam(net.parameters(), lr,
                                weight_decay=1e-5, amsgrad=True)

    # training
    def train_update(engine, batch):
        net.train()
        optimizer.zero_grad()
        x, y, batch = prepare_batch(batch, device, args)
        y_pred = net(x)
        if args.model == 'unet':
            loss = loss_fn(y_pred, y)
        elif args.model == 'local_unet':
            loss = loss_fn(y_pred, y, mask=batch['mask'])
        loss.backward()
        optimizer.step()
        return prepare_output(batch, y, y_pred, loss)
    trainer = Engine(train_update)

    # validation
    def val_inference(engine, batch):
        net.eval()
        with torch.no_grad():
            x, y, batch = prepare_batch(batch, device, args)
            y_pred = net(x)
            return prepare_output(batch, y, y_pred, None)
    evaluator = Engine(val_inference)
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def run_validation(engine):
        evaluator.run(val_loader)

    # metrics
    trlm = ignite.metrics.Loss(loss_fn,
                  output_transform=lambda out: (out["y_pred"], out["y"]))
    trlm.attach(trainer, "tr_loss")
    vallm = ignite.metrics.Loss(loss_fn,
                  output_transform=lambda out: (out["y_pred"], out["y"]))
    vallm.attach(evaluator, "val_loss")
    trmd = MeanDice(sigmoid=True,
                  output_transform=lambda out: (out["y_pred"], out["y"]))
    trmd.attach(trainer, "tr_Mean_Dice")
    valmd = MeanDice(sigmoid=True,
                  output_transform=lambda out: (out["y_pred"], out["y"]))
    valmd.attach(evaluator, "val_Mean_Dice")
    trd1 = MeanDice(sigmoid=True,
         output_transform=lambda out: (out["y_pred"][:, 0:1], out["y"][:, 0:1]))
    trd1.attach(trainer, "tr_Dice_ET")
    vald1 = MeanDice(sigmoid=True,
         output_transform=lambda out: (out["y_pred"][:, 0:1], out["y"][:, 0:1]))
    vald1.attach(evaluator, "val_Dice_ET")
    trd2 = MeanDice(sigmoid=True,
         output_transform=lambda out: (out["y_pred"][:, 1:2], out["y"][:, 1:2]))
    trd2.attach(trainer, "tr_Dice_TC")
    vald2 = MeanDice(sigmoid=True,
         output_transform=lambda out: (out["y_pred"][:, 1:2], out["y"][:, 1:2]))
    vald2.attach(evaluator, "val_Dice_TC")
    trd3 = MeanDice(sigmoid=True,
         output_transform=lambda out: (out["y_pred"][:, 2:3], out["y"][:, 2:3]))
    trd3.attach(trainer, "tr_Dice_WT")
    vald3 = MeanDice(sigmoid=True,
         output_transform=lambda out: (out["y_pred"][:, 2:3], out["y"][:, 2:3]))
    vald3.attach(evaluator, "val_Dice_WT")

    # checkpoints
    checkpoint_handler = ModelCheckpoint(checkpoint_dir, "net",
                global_step_transform=lambda eng, ev_name: trainer.state.epoch,
                n_saved=None,
                require_empty=False)
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED(every=20),
        handler=checkpoint_handler,
        to_save={"net": net, "opt": optimizer},
    )

    # logging
    # to console
    stats_handler = StatsHandler(
        name="logger",
        output_transform=lambda out: out["loss"],
    )
    stats_handler.attach(trainer)
    stats_handler.attach(evaluator)
    # to tensorboard
    tensorboard_stats_handler = TensorBoardStatsHandler(
        log_dir=log_dir,
        output_transform=lambda out: out["loss"],
        global_epoch_transform=lambda x: trainer.state.epoch,
    )
    tensorboard_stats_handler.attach(trainer)
    tensorboard_stats_handler.attach(evaluator)

    ## add handler to draw the first image and the corresponding label and model output in the last batch
    ## here we draw the 3D output as GIF format along the depth axis, every 2 validation iterations.
    #val_tensorboard_image_handler = TensorBoardImageHandler(
    #    batch_transform=lambda batch: (batch["img"], batch["seg"]),
    #    output_transform=lambda output: predict_segmentation(output[0]),
    #    global_iter_transform=lambda x: trainer.state.epoch,
    #)
    #evaluator.add_event_handler(event_name=Events.ITERATION_COMPLETED(every=2), handler=val_tensorboard_image_handler)

    state = trainer.run(train_loader, args.train_epochs)
    print(state)


if __name__ == "__main__":
    args = options.parser.parse_args()
    main(args)
