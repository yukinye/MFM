import yaml
import string
import secrets
import os

import torch
import wandb
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torchdyn.core import NeuralODE

from mfm.utils import plot_images_trajectory
from mfm.networks.utils import flow_model_torch_wrapper


def load_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def merge_config(args, config_updates):
    for key, value in config_updates.items():
        if not hasattr(args, key):
            raise ValueError(
                f"Unknown configuration parameter '{key}' found in the config file."
            )
        setattr(args, key, value)
    return args


def generate_group_string(length=16):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def dataset_name2datapath(dataset_name, working_dir):
    if dataset_name == "eb":
        return os.path.join(working_dir, "data", "eb_velocity_v5.npz")
    elif dataset_name == "cite":
        return os.path.join(working_dir, "data", "op_cite_inputs_0.h5ad")
    elif dataset_name == "multi":
        return os.path.join(working_dir, "data", "op_train_multi_targets_0.h5ad")
    elif dataset_name == "lidar":
        return os.path.join(working_dir, "data", "rainier2-thin.las")
    elif dataset_name == "afhq":
        return os.path.join(working_dir, "data", "afhq")
    elif dataset_name == "celeba":
        return os.path.join(working_dir, "data", "celeba")
    else:
        raise ValueError("Dataset not recognized")


def create_callbacks(args, phase, data_type, run_id, datamodule=None):

    dirpath = os.path.join(
        args.working_dir,
        "checkpoints",
        data_type,
        str(run_id),
        f"{phase}_model",
    )

    if phase == "geopath":
        early_stop_callback = EarlyStopping(
            monitor="GeoPathNet/val_loss_geopath",
            patience=args.patience_geopath,
            mode="min",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="GeoPathNet/val_loss_geopath",
            mode="min",
            save_top_k=1,
        )
        callbacks = [checkpoint_callback, early_stop_callback]
    elif phase == "flow":
        if args.data_type == "image":
            checkpoint_callback = ModelCheckpoint(
                dirpath=dirpath,
                every_n_epochs=args.check_val_every_n_epoch,
            )
            plotting_callback = PlottingCallback(
                plot_interval=args.check_val_every_n_epoch, datamodule=datamodule
            )
            callbacks = [checkpoint_callback, plotting_callback]
        else:
            early_stop_callback = EarlyStopping(
                monitor="FlowNet/val_loss_cfm",
                patience=args.patience,
                mode="min",
            )
            checkpoint_callback = ModelCheckpoint(
                dirpath=dirpath,
                mode="min",
                save_top_k=1,
            )
            callbacks = [checkpoint_callback, early_stop_callback]
    else:
        raise ValueError("Unknown phase")
    return callbacks


class PlottingCallback(Callback):
    def __init__(self, plot_interval, datamodule):
        self.plot_interval = plot_interval
        self.datamodule = datamodule

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        pl_module.flow_net.train(mode=False)
        if epoch % self.plot_interval == 0 and epoch != 0:
            node = NeuralODE(
                flow_model_torch_wrapper(pl_module.flow_net).to(self.datamodule.device),
                solver="tsit5",
                sensitivity="adjoint",
                atol=1e-5,
                rtol=1e-5,
            )

            for mode in ["train", "val"]:
                x0 = getattr(self.datamodule, f"{mode}_x0")
                x0 = x0[0:15]
                fig = self.trajectory_and_plot(x0, node, self.datamodule)
                wandb.log({f"Trajectories {mode.capitalize()}": wandb.Image(fig)})
        pl_module.flow_net.train(mode=True)

    def trajectory_and_plot(self, x0, node, datamodule):
        selected_images = x0[0:15]
        with torch.no_grad():
            traj = node.trajectory(
                selected_images.to(datamodule.device),
                t_span=torch.linspace(0, 1, 100).to(datamodule.device),
            )

        traj = traj.transpose(0, 1)
        traj = traj.reshape(*traj.shape[0:2], *datamodule.dim)

        fig = plot_images_trajectory(
            traj.to(datamodule.device),
            datamodule.vae.to(datamodule.device),
            datamodule.process,
            num_steps=5,
        )
        return fig
