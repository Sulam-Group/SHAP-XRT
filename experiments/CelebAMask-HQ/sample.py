import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import datasets
import sde_lib
import sampling
from absl import app, flags
from ml_collections import config_flags
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image
from models import utils as mutils
from models.ema import ExponentialMovingAverage

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", "configs/celebamaskhq_256.py", "Configuration", lock_config=True
)
flags.DEFINE_string("gpu", "0,1,2,3,4,5,6,7", "Device(s) to use")
flags.DEFINE_string("workdir", "./", "Working directory")


def setup(device, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=device, world_size=world_size)
    torch.cuda.set_device(device)


def sample(rank, world_size, config, workdir):
    device = rank
    setup(device, world_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    sample_dir = os.path.join(workdir, "sample")
    os.makedirs(sample_dir, exist_ok=True)

    # Initialize score network
    from models import ncsnpp

    model_name = config.model.name
    score_model = mutils.get_model(model_name)(config)

    checkpoint = torch.load(
        os.path.join(checkpoint_dir, "score.pt"), map_location=device
    )
    state_dict = checkpoint["module"]
    score_model.load_state_dict(state_dict)
    score_model = score_model.to(device)
    score_model.eval()
    score_model = DDP(score_model, device_ids=[device], output_device=device)

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )

    # Initialize dataset and distributed dataloader
    _, dataset = datasets.get_dataset(config)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=4)
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Building sampling functions
    sampling_batch_size = 16
    if config.training.snapshot_sampling:
        sampling_shape = (
            sampling_batch_size,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size,
        )
        sampling_fn = sampling.get_sampling_fn(
            config, sde, sampling_shape, inverse_scaler, sampling_eps
        )

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    sample, n = sampling_fn(score_model)
    ema.restore(score_model.parameters())

    nrow = int(np.sqrt(sample.size(0)))
    image_grid = make_grid(sample, nrow, padding=2)
    save_image(image_grid, os.path.join(sample_dir, f"sample_{rank}.png"))


def main(_):
    config = FLAGS.config
    workdir = FLAGS.workdir
    gpu = FLAGS.gpu

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    world_size = len(gpu.split(","))
    mp.spawn(sample, args=(world_size, config, workdir), nprocs=world_size)


if __name__ == "__main__":
    app.run(main)
