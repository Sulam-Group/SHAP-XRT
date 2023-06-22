import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import datasets
import sde_lib
import repaint_lib
import itertools
from absl import app, flags
from ml_collections import config_flags
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from models import utils as mutils
from tqdm import tqdm

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


def repaint(rank, world_size, config, workdir):
    device = rank
    setup(device, world_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    repaint_dir = os.path.join(workdir, "repaint")

    # Initialize dataset and distributed dataloader
    _, dataset = datasets.get_dataset(config)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=4)
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)

    from models import ncsnpp

    # Initialize score network
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
    torch.set_grad_enabled(False)

    # Setup repaint schedule
    sigma_min, sigma_max = config.model.sigma_min, config.model.sigma_max
    repaint_steps = 2000
    repaint_jump = 10
    repaint_n_samples = 7
    repaint_schedule = repaint_lib.get_schedule(
        1, repaint_steps, repaint_jump, repaint_n_samples, eps=1e-05
    )
    sigmas = sigma_min * (sigma_max / sigma_min) ** repaint_schedule
    sigmas = sigmas.to(device)

    # Setup feature subsets
    gamma = 5
    s = repaint_lib.get_s(gamma)

    sampling_batch_size = 16
    for data in dataloader:
        idx, x0, attr, feature = data

        attr_name = "Smiling"
        attr_idx = dataset.attr_names.index(attr_name)
        target = attr[:, attr_idx].item()
        if not target:
            continue
        idx = idx.item()

        idx_repaint_dir = os.path.join(repaint_dir, f"{idx}")
        os.makedirs(idx_repaint_dir, exist_ok=True)

        torch.save(x0, os.path.join(idx_repaint_dir, f"original.pt"))

        x0 = x0.to(device)
        x0 = scaler(x0)

        x0 = x0.repeat(sampling_batch_size, 1, 1, 1)

        for _s in tqdm(s[::-1]):
            s_repaint_dir = os.path.join(idx_repaint_dir, "".join([str(u) for u in _s]))
            s_checkpoint_repaint_dir = os.path.join(s_repaint_dir, "checkpoints")
            os.makedirs(s_checkpoint_repaint_dir, exist_ok=True)

            m = repaint_lib.get_mask(feature, _s)
            m = m.to(device)
            torch.save(m.cpu(), os.path.join(s_repaint_dir, f"mask.pt"))

            m = m.repeat(sampling_batch_size, 1, 1, 1)
            x = repaint_lib.repaint(
                x0, m, score_model, sigmas, s_checkpoint_repaint_dir
            )
            torch.save(x.cpu(), os.path.join(s_repaint_dir, f"repainted.pt"))


def main(_):
    config = FLAGS.config
    workdir = FLAGS.workdir
    gpu = FLAGS.gpu

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    world_size = len(gpu.split(","))
    mp.spawn(repaint, args=(world_size, config, workdir), nprocs=world_size)


if __name__ == "__main__":
    app.run(main)
