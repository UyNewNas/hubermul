"""Independent seeded splits; target contamination and observation contamination differ."""

import torch


def make_split(task, seed, size, contamination, outlier_scale, extrapolation_scale):
    g = torch.Generator().manual_seed(seed)
    clean = torch.rand(size, 4, generator=g) * 2 - 1
    if task == "corrupted_observation":
        mask = torch.rand(size, 1, generator=g) < contamination
        challenge = clean + mask * torch.randn(size, 4, generator=g) * outlier_scale
        target_input = clean
    elif task == "legitimate_large":
        mask = torch.rand(size, 1, generator=g) < contamination
        challenge = clean * torch.where(mask, outlier_scale, 1.0)
        target_input = challenge
    else:
        challenge = clean * extrapolation_scale
        target_input = challenge

    def target(x):
        return (x[:, 0] + x[:, 1]) * (x[:, 2] - x[:, 3])

    return clean, target(clean), challenge, target(target_input)
