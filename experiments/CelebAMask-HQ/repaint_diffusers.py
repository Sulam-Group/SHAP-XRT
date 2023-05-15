import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import repaint_lib
import itertools
from absl import app, flags
from diffusers import DiffusionPipeline
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset import get_dataset
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("gpu", "0,1,2,3,4,5,6,7", "Device(s) to use")
flags.DEFINE_string("workdir", "./", "Working directory")


def setup(device, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=device, world_size=world_size)
    torch.cuda.set_device(device)


def repaint(rank, world_size, workdir):
    device = rank
    setup(device, world_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repaint_dir = os.path.join(workdir, "repaint")

    dataset = get_dataset(
        "val",
        mu=torch.tensor([0.0, 0.0, 0.0]),
        std=torch.tensor([1.0, 1.0, 1.0]),
        return_feature=True,
        return_idx=True,
    )
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=4)

    model_id = "google/ncsnpp-celebahq-256"
    sde_ve = DiffusionPipeline.from_pretrained(model_id)
    scheduler, unet = sde_ve.scheduler, sde_ve.unet

    sigma_min, sigma_max = scheduler.config.sigma_min, scheduler.config.sigma_max
    repaint_steps = 2000
    repaint_jump = 10
    repaint_n_samples = 7
    repaint_schedule = repaint_lib.get_schedule(
        1,
        repaint_steps,
        repaint_jump,
        repaint_n_samples,
        eps=scheduler.config.sampling_eps,
    )
    sigmas = sigma_min * (sigma_max / sigma_min) ** repaint_schedule
    sigmas = sigmas.to(device)

    unet = unet.to(device)
    unet.eval()
    unet = DDP(unet, device_ids=[device], output_device=device)
    torch.set_grad_enabled(False)

    gamma = 5
    s = repaint_lib.get_s(gamma=gamma)

    sampling_batch_size = 16
    for data in dataloader:
        idx, x0, attr, feature = data

        attr_name = "Smiling"
        attr_idx = dataset.attr_names.index(attr_name)
        target = attr[:, attr_idx].item()
        if not target:
            continue
        idx = idx.item()

        idx_repaint_dir = os.path.join(repaint_dir, str(idx))
        os.makedirs(idx_repaint_dir, exist_ok=True)

        torch.save(x0, os.path.join(idx_repaint_dir, "original.pt"))

        x0 = x0.to(device)
        feature = feature.to(device)

        x0 = x0.repeat(sampling_batch_size, 1, 1, 1)

        for _s in tqdm(s[::-1]):
            s_repaint_dir = os.path.join(idx_repaint_dir, "".join([str(u) for u in _s]))
            s_checkpoint_repaint_dir = os.path.join(s_repaint_dir, "checkpoints")
            os.makedirs(s_checkpoint_repaint_dir, exist_ok=True)

            m = repaint_lib.get_mask(feature, _s)
            torch.save(m.cpu(), os.path.join(s_repaint_dir, f"mask.pt"))

            m = m.repeat(sampling_batch_size, 1, 1, 1)
            x = repaint_lib.repaint(x0, m, unet, sigmas, s_checkpoint_repaint_dir)

            torch.save(x.cpu(), os.path.join(s_repaint_dir, "repainted.pt"))


def main(_):
    workdir = FLAGS.workdir
    gpu = FLAGS.gpu

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    world_size = len(gpu.split(","))
    mp.spawn(repaint, args=(world_size, workdir), nprocs=world_size)


if __name__ == "__main__":
    app.run(main)
