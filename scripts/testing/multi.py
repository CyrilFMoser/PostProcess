import os
import socket
import torch
import torch.distributed as dist

def main():
    # Init process group (torchrun handles env vars)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    hostname = socket.gethostname()

    # Each rank contributes a unique value
    # Example: rank 0 -> 1, rank 1 -> 2, ...
    local_value = torch.tensor([rank + 1.0], device=device)

    # Clone for reduction
    reduced_value = local_value.clone()

    # Synchronize before reduction
    #dist.barrier()

    # All-reduce SUM across *all nodes*
    dist.all_reduce(reduced_value, op=dist.ReduceOp.SUM)

    # Synchronize after reduction
    dist.barrier()

    if rank == 0:
        expected = world_size * (world_size + 1) / 2
        print("=" * 60)
        print(f"WORLD SIZE: {world_size}")
        print(f"EXPECTED SUM: {expected}")
        print("=" * 60)

    print(
        f"[Rank {rank:03d} | Local rank {local_rank} | {hostname}] "
        f"local={local_value.item():.1f} "
        f"reduced={reduced_value.item():.1f}"
    )

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
