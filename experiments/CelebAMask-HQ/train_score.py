import os
import torch
import torchvision.transforms as t
import deepspeed as ds
import sde_lib
import losses
from absl import app, flags
from ml_collections import config_flags
from dataset import get_dataset
from models import ncsnpp
from models import utils as mutils

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", "configs/celebamaskhq_256.py", "Configuration", lock_config=True
)
flags.DEFINE_string("include", None, "Device(s) to include")
flags.DEFINE_integer("local_rank", -1, "Local rank passed from distributed launcher")
flags.DEFINE_integer("master_port", 29500, "Master port for distributed training")
flags.DEFINE_string("workdir", "./", "Working directory")


def main(_):
    config = FLAGS.config

    augmentation = t.Compose(
        [t.RandomVerticalFlip(), t.RandomHorizontalFlip(), t.RandomRotation(180)]
    )
    dataset = get_dataset(
        "train",
        mu=torch.tensor([0.0, 0.0, 0.0]),
        std=torch.tensor([1.0, 1.0, 1.0]),
        augmentation=augmentation,
    )

    model = mutils.get_model(config.model.name)(config)
    model_engine, _, train_loader, _ = ds.initialize(
        args=config,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
    )

    checkpoint_dir = os.path.join(FLAGS.workdir, "checkpoints", "score")

    sde = sde_lib.VESDE(
        sigma_min=config.model.sigma_min,
        sigma_max=config.model.sigma_max,
        N=config.model.num_scales,
    )

    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    loss_fn = losses.get_sde_loss_fn(
        sde,
        train=True,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )

    step = 0
    model_engine.save_checkpoint(checkpoint_dir, tag=f"step_{step}")
    while step < config.training.n_iters:
        for data in train_loader:
            x, _ = data

            x = x.to(model_engine.device)
            loss = loss_fn(model_engine, x)

            model_engine.backward(loss)
            model_engine.step()
            step += 1

            if (step % config.training.snapshot_freq) == 0:
                model_engine.save_checkpoint(checkpoint_dir, tag=f"step_{step}")
    model_engine.save_checkpoint(checkpoint_dir, tag="final")


if __name__ == "__main__":
    app.run(main)
