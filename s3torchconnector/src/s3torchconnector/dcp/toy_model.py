import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
from torch import nn
import argparse

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

from s3torchconnector import S3StorageWriter, S3StorageReader

CHECKPOINT_DIR = "checkpoint"


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(backend):
    dist.init_process_group(backend)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_save_example(rank, backend):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")

    if backend == "nccl":
        # Need to put tensor on a GPU device for nccl backend
        device_id = rank % torch.cuda.device_count()
        model = ToyModel().to(device_id)
        model = FSDP(model, device_id=device_id)
    elif backend == "gloo":
        device_id = torch.device("cpu")
        model = ToyModel().to(device_id)
        model = FSDP(model)
    else:
        raise Exception(f"Unknown backend type: {backend}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    model(torch.rand(8, 16).to(device_id)).sum().backward()
    optimizer.step()

    loaded_state_dict = {}
    # DCP.load(
    #     loaded_state_dict,
    #     # storage_reader=DCP.FileSystemReader(CHECKPOINT_DIR)
    #     storage_reader=S3StorageReader(region="eu-north-1", s3_uri="s3://dcp-poc-test/", thread_count=world_size)
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
        storage_writer=S3StorageWriter(
            region="eu-north-1", s3_uri="s3://dcp-poc-test/epoech_1/"
        ),
    )

    state_dict = {
        "model": model.state_dict(),
        "prefix": "bla",
    }
    optimizer.step()

    DCP.save(
        state_dict=state_dict,
        # storage_writer=DCP.FileSystemWriter(CHECKPOINT_DIR),
        storage_writer=S3StorageWriter(
            region="eu-north-1", s3_uri="s3://dcp-poc-test/epoech_2/"
        ),
    )


if __name__ == "__main__":
    """
    How to use:
    Step 1: Set up EC2 Instances
        Create two EC2 instances on AWS.
        Modify the security groups of these instances to allow inbound TCP/UDP connections on all ports between them.

    Step 2: Designate Master and Worker Hosts. Choose one instance as the master host. In this example, we'll assume
    the master host's IP address is 172.31.18.217. Decide on the port number the master host will listen on
    for the worker host. For this guide, we'll use port 1234.

    Step 3: Run Commands on Master and Worker Hosts. For CPU training, run the following commands simultaneously
    on the master and worker hosts:
    Master Host:
        torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=172.31.18.217 --master_port=1234 toy_model.py --backend=gloo
    Worker Host:
        torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=172.31.18.217 --master_port=1234 toy_model.py --backend=gloo

    Here's what each argument means:
        --nproc_per_node=4: Run four processes on each node (instance).
        --nnodes=2: Use two nodes (instances) for training.
        --node_rank=0 (master) / --node_rank=1 (worker): Set the rank of the current node.
        --master_addr=172.31.18.217: Set the IP address of the master host.
        --master_port=1234: Set the port number the master host is listening on.
        toy_model.py: The script to run for training.
        --backend=gloo: Use the gloo backend, which utilizes CPU for training.

    For GPU training, run the following commands simultaneously on the master and worker hosts:
    Master Host:
        torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=172.31.18.217 --master_port=1234 toy_model.py --backend=nccl
    Worker Host:
        torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=172.31.18.217 --master_port=1234 toy_model.py --backend=nccl

    The meaningful difference from the CPU training commands is the --backend=nccl argument, which uses the nccl
    backend for GPU training. Also, nproc_per_node was set to 1 to support running that command on instance with only
    one GPU. When you use GPU for training,you need to limit amount of processes per node (nproc_per_node),
    by amount of GPU available on node.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    args = parser.parse_args()

    setup(args.backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Starting for rank {rank}, world_size is {world_size}")

    run_fsdp_checkpoint_save_example(rank, args.backend)

    cleanup()
