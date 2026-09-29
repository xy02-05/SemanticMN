#!/usr/bin/env python3

import os

import torch
import torch.distributed as dist

from openpi.models import pi0_config
from openpi.training import data_loader


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    global_batch_size = 32

    dataset = data_loader.FakeDataset(
        pi0_config.Pi0Config(action_dim=32, action_horizon=50, max_token_len=48),
        num_samples=256,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=global_batch_size // world_size,
        sampler=sampler,
        num_batches=1,
        framework="pytorch",
    )
    batch = next(iter(loader))
    local_batch_size = int(batch["actions"].shape[0])
    sizes = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(sizes, torch.tensor([local_batch_size], dtype=torch.long))
    observed_global_batch_size = sum(int(size.item()) for size in sizes)
    if local_batch_size != 16 or observed_global_batch_size != global_batch_size:
        raise AssertionError(
            f"rank={rank} local={local_batch_size} global={observed_global_batch_size}"
        )
    print(
        f"batch_ok rank={rank} local_batch={local_batch_size} "
        f"world_size={world_size} global_batch={observed_global_batch_size} "
        f"local_rank={os.environ.get('LOCAL_RANK')}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
