import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import sde_lib
import repaint_lib
import itertools
from absl import app, flags
from ml_collections import config_flags
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset import get_dataset
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

    checkpoint_dir = os.path.join(workdir, "checkpoints", "score", "step_300000")
    repaint_dir = os.path.join(workdir, "repaint")

    dataset = get_dataset(
        "train",
        mu=torch.tensor([0.0, 0.0, 0.0]),
        std=torch.tensor([1.0, 1.0, 1.0]),
        return_feature=True,
        return_idx=True,
    )
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=4)

    from models import ncsnpp

    model = mutils.get_model(config.model.name)(config)
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt"), map_location=device
    )
    state_dict = checkpoint["module"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model = DDP(model, device_ids=[device], output_device=device)
    torch.set_grad_enabled(False)

    sigma_min, sigma_max = config.model.sigma_min, config.model.sigma_max
    sde = sde_lib.VESDE(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        N=config.model.num_scales,
    )

    train = False
    continuous = config.training.continuous
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)

    rsde = sde.reverse(score_fn)

    N = 250
    j = r = 10
    sigma0 = 1.0
    T = (np.log(sigma0) - np.log(sigma_min)) / (np.log(sigma_max) - np.log(sigma_min))
    repaint_schedule = repaint_lib.get_schedule(T, N, j, r)
    repaint_schedule = repaint_schedule.to(device)

    gamma = 5
    s = []
    for c in range(1, gamma):
        s.extend(list(itertools.combinations(range(gamma), c)))

    sampling_batch_size = 32
    for data in dataloader:
        idx, x0, _, feature = data
        idx = idx.item()

        idx_repaint_dir = os.path.join(repaint_dir, f"{idx}")
        os.makedirs(idx_repaint_dir, exist_ok=True)

        torch.save(x0, os.path.join(idx_repaint_dir, f"original.pt"))

        x0 = x0.to(device)
        feature = feature.to(device)

        for _s in tqdm(s[::-1]):
            _s = [2, 3]
            s_repaint_dir = os.path.join(idx_repaint_dir, "".join([str(u) for u in _s]))
            s_checkpoint_repaint_dir = os.path.join(s_repaint_dir, "checkpoints")
            os.makedirs(s_checkpoint_repaint_dir, exist_ok=True)

            m = repaint_lib.get_mask(feature, _s)

            torch.save(m.cpu(), os.path.join(s_repaint_dir, f"mask.pt"))

            x0 = x0.repeat(sampling_batch_size, 1, 1, 1)
            m = m.repeat(sampling_batch_size, 1, 1, 1)
            x = repaint_lib.repaint(
                x0, m, sde, rsde, repaint_schedule, s_checkpoint_repaint_dir
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
