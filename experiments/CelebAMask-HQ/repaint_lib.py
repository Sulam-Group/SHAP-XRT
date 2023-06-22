import os
import torch
import itertools
from skimage.morphology import disk, erosion
from tqdm import tqdm


def get_s(gamma=5):
    s = []
    for c in range(1, gamma):
        s.extend(list(itertools.combinations(range(gamma), c)))
    return s


def get_schedule(T, N, j, r, eps=1e-5):
    jumps = {}
    for t in range(0, N - j, j):
        jumps[t] = r - 1

    t = N
    schedule = [N]
    while t > 0:
        t -= 1
        schedule.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(r):
                t += 1
                schedule.append(t)

    schedule = torch.tensor(schedule)
    schedule = eps + (T - eps) * schedule / N
    schedule = schedule.unsqueeze(1)
    return schedule


def get_mask(feature, s):
    s = torch.tensor(s)
    s = s.to(feature.device)
    m = torch.isin(feature, s).float()

    footprint = disk(2)
    m = m.squeeze()
    m = erosion(m, footprint)
    m = torch.from_numpy(m)
    return m


def repaint(x0, m, score_model, sigmas, checkpoint_dir):
    # initialize random noise
    sigma0 = sigmas[0]
    z = torch.randn_like(x0)
    x = sigma0[:, None, None, None] * z

    torch.save(x.cpu(), os.path.join(checkpoint_dir, f"0.pt"))

    def _reverse_step(x, sigma, sigma_next):
        # add noise to known part
        z = torch.randn_like(x)
        x_known = x0 + sigma_next[:, None, None, None] * z

        # denoise unknown part
        model_output = score_model(x, sigma.repeat(x.size(0)))

        diffusion = torch.sqrt(sigma**2 - sigma_next**2)
        diffusion = diffusion.flatten()
        drift = -(diffusion**2) * model_output

        z = torch.randn_like(x)
        x_unknwon = x - drift + diffusion * z

        x = m * x_known + (1 - m) * x_unknwon
        return x

    def _forward_step(x, sigma, sigma_next):
        # add noise
        z = torch.randn_like(x)
        diffusion = torch.sqrt(sigma_next**2 - sigma**2)
        x = x + diffusion[:, None, None, None] * z
        return x

    for i, sigma in enumerate(tqdm(sigmas[:-1])):
        sigma_next = sigmas[i + 1]
        if sigma_next < sigma:
            x = _reverse_step(x, sigma, sigma_next)
        else:
            x = _forward_step(x, sigma, sigma_next)

        checkpoint_step = 500
        if (i + 1) % checkpoint_step == 0:
            torch.save(x.cpu(), os.path.join(checkpoint_dir, f"{i + 1}.pt"))

    return x
