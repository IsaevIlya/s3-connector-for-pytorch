import os
from multiprocessing import Process

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
import torch.multiprocessing as mp
from torch import nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

from s3torchconnector import S3StorageWriter

CHECKPOINT_DIR = "checkpoint"

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["GLOO_SOCKET_IFNAME"] = "en0"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cpu.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_save_example(rank, world_size):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(device=torch.device("cpu"))
    model = FSDP(model, device_id=torch.cpu.current_device())

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    model(torch.rand(8, 16, device=torch.device("cpu"))).sum().backward()
    optimizer.step()

    loaded_state_dict = {}
    # DCP.load(
    #     loaded_state_dict,
    #     storage_reader=DCP.FileSystemReader(CHECKPOINT_DIR)
    # )

    # set FSDP StateDictType to SHARDED_STATE_DICT so we can use DCP to checkpoint sharded model state dict
    # note that we do not support FSDP StateDictType.LOCAL_STATE_DICT
    FSDP.set_state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
    )
    state_dict = {
        "model": model.state_dict(),
    }

    DCP.save(
        state_dict=state_dict,
        # storage_writer=DCP.FileSystemWriter(CHECKPOINT_DIR, single_file_per_rank=True),
        storage_writer=S3StorageWriter(region="eu-north-1", s3_uri="s3://dcp-poc-test/"),
    )

    state_dict = {
        "model": model.state_dict(), "prefix": "bla",
    }
    optimizer.step()

    DCP.save(
        state_dict=state_dict,
        # storage_writer=DCP.FileSystemWriter(CHECKPOINT_DIR),
        storage_writer=S3StorageWriter(region="eu-north-1", s3_uri="s3://dcp-poc-test/"),
    )

    cleanup()


if __name__ == "__main__":
    world_size = 8
    # world_size = 1
    print(f"Running fsdp checkpoint example on {world_size} devices.")
    mp.set_start_method('fork')
    p = Process(target = run_fsdp_checkpoint_save_example, )
    mp.spawn(
        run_fsdp_checkpoint_save_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
