#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import logging
import functools
from dataclasses import dataclass
from time import perf_counter
from typing import Tuple
import os
import argparse

import torch.distributed.checkpoint as dcp
import torch
import torch.distributed as dist
import torch.utils.data

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader

from s3torchconnector.dcp import S3StorageWriter, S3StorageReader


from s3torchbenchmarking.models import get_benchmark_model


from s3torchbenchmarking.benchmark_utils import (
    build_random_suffix,
    build_checkpoint_uri,
)

from s3torchconnector.s3reader import S3ReaderConstructor

Timestamps = Tuple[float, float]
logger = logging.getLogger(__name__)

import sys
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

from s3torchconnector import S3ReaderConstructor

def setup(backend: str, world_size: int, rank: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    dist.init_process_group(backend, world_size=world_size, rank=rank)


def get_writer(region:str, uri: str, suffix: str, thread_count: int = 8) -> FileSystemWriter:
    uri = build_checkpoint_uri(uri, suffix)
    logger.info("Saving checkpoint to %s (S3)...", uri)
    return S3StorageWriter(region, uri, thread_count=thread_count)

def get_reader(region:str, uri: str, suffix: str) -> FileSystemReader:
    uri = build_checkpoint_uri(uri, suffix)
    logger.info("Loading checkpoint from %s (S3)...", uri)
    # reader_constructor = S3ReaderConstructor.sequential()
    reader_constructor = S3ReaderConstructor.range_based(1*1024*1024)
    return S3StorageReader(region, uri, reader_constructor=reader_constructor)

import re
# from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

class VirtualRegexContainer:
    def __init__(self, regex: str):
        self.regex = re.compile(regex)

    def __contains__(self, item: str) -> bool:
        # print(f"Item name: {item}")
        verdict = self.regex.search(item) is not None
        if not verdict:
            print(f"Skipping {item}")
        return verdict


from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint._traverse import set_element
from typing import List, Optional



class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
    """
    Extension of DefaultLoadPlanner, which rebuilds state_dict from the saved metadata.
    Useful for loading in state_dict without first initializing a model, such as
    when converting a DCP checkpoint into a Torch save file.

    . N.B. `state_dict` must be an empty dictionary when used with this LoadPlanner

    .. warning::
        Because the entire state dict is initialized, It's recommended to only utilize
        this LoadPlanner on a single rank or process to avoid OOM.

    """

    def __init__(self, keys=None, *args, **kwargs):
        self.keys = keys
        super().__init__(*args, **kwargs)

    def _should_include_key(self, key: str, metadata: Metadata) -> bool:
        if self.keys is None:
            return True

        if key in self.keys:
            True

        unflattened_keys: List[str] = []
        planner_data = metadata.planner_data.get(key)
        for unflattened_key in planner_data:
            if unflattened_keys:
                unflattened_keys.append(
                    ".".join([unflattened_keys[-1], str(unflattened_key)])
                )

            else:
                unflattened_keys.append(unflattened_key)

        if any(unflattened_key in self.keys for unflattened_key in unflattened_keys):
            return True

        return False

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Optional[Metadata] = None,
        is_coordinator: bool = False,
    ) -> None:
        assert not state_dict
        assert metadata is not None

        # rebuild the state dict from the metadata
        for k, v in metadata.state_dict_metadata.items():
            if not self._should_include_key(k, metadata):
                continue

            if isinstance(v, TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            if k in metadata.planner_data:
                set_element(state_dict, metadata.planner_data[k], v)
            else:
                state_dict[k] = v

        super().set_up_planner(state_dict, metadata, is_coordinator)



def run_fsdp(
    rank: int,  # needs to be passed first (provided by `multiprocessing.spawn` automatically)
    world_size: int,
    thread_count: int,
    backend: str,
    region: str,
    uri: str,
    suffix: str,
    model_name: str = "L7b",
    checkpoint_sharding_strategy: str = "full"
) -> None:
    """Execute the actual code for checkpoint saving.

    This function is meant to be executed in subprocesses."""
    # setup(backend=backend, world_size=world_size, rank=rank)

    if rank == 0:
        logger.info("Creating Model")
    # Instantiate model on CPU on rank=0 only to prevent CPU OOM
    # (e.g. 70B * 4 bytes * 8 processes > 2T RAM available on P5)
    if rank == 0:
        model_proxy = get_benchmark_model(model_name)
        model = model_proxy.model
    else:
        with torch.device("meta"):
            # Instantiating model on `meta` device doesn't consume CPU memory,
            # but requires specifing `param_init_fn=...`
            # and `sync_module_states=True` in FSDP c-tor.
            model_proxy = get_benchmark_model(model_name)
            model = model_proxy.model

    model_size = model_proxy.size
    model_name = model_proxy.name
    if rank == 0:
        logger.info(f"Model {model_name} created")

    transformer_layer = LlamaDecoderLayer
    gpt_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            transformer_layer,
        },
    )

    if backend == "nccl":
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        param_init_fn = lambda module: module.to_empty(
            device=torch.device("cuda"), recurse=False
        )
    else:
        device_id = rank % torch.cpu.device_count()
        torch.cpu.set_device(device_id)
        param_init_fn = lambda module: module.to_empty(
            device=torch.device("cpu"), recurse=False
        )

    # =======================================
    # if checkpoint_sharding_strategy == "full":
    #     sharding_strategy = ShardingStrategy.FULL_SHARD
    # elif checkpoint_sharding_strategy == "hybrid":
    #     sharding_strategy = ShardingStrategy.HYBRID_SHARD
    # else:
    #     raise NotImplementedError("Available sharding strategies are full and hybrid")
    #
    # model = FSDP(
    #     model,
    #     auto_wrap_policy=gpt_auto_wrap_policy,
    #     device_id=(
    #         torch.cuda.current_device()
    #         if backend == "nccl"
    #         else torch.cpu.current_device()
    #     ),
    #     use_orig_params=False,
    #     sharding_strategy=sharding_strategy,
    #     sync_module_states=True if backend == "nccl" else False,
    #     param_init_fn=param_init_fn if rank != 0 else None,
    # )
    #
    # if rank == 0:
    #     print("Wrapped model with FSDP")
    #
    # # torch.cuda.empty_cache()
    # with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    #     state_dict = {
    #         "model": model.state_dict(),
    #     }
    #
    # storage_writer = get_writer(region, uri, suffix, thread_count)
    # # align all workers to start checkpointing at the same time
    # dist.barrier()
    # begin_save = perf_counter()
    # dcp.save(state_dict, storage_writer=storage_writer)
    #
    # dist.barrier()
    # end_save = perf_counter()
    #
    # if rank == 0:
    #     print(f"The total size of model is {model_size}")
    #     print(f"Time taken to save: {end_save - begin_save} seconds")
    # # Record the save times excluding the influence of the process setup and model loading to device.
    # return



    storage_reader = get_reader(region, uri, suffix)
    # empty_stat_dict = {"model": None}
    start_load = perf_counter()
    # dcp.load(state_dict, storage_reader=storage_reader)
    model_only = True
    sd_out: STATE_DICT_TYPE = {}
    #     {
    #     "model": None
    # }
    # keys_regex = None if not model_only else VirtualRegexContainer("^model\\.*")
    keys_regex = None if not model_only else VirtualRegexContainer("^model\\.model\\.layers\\.[13][13579]")
    load_planner = _EmptyStateDictLoadPlanner(keys=keys_regex)
    _load_state_dict(
        sd_out,
        storage_reader,
        planner=load_planner,
        no_dist=True,
    )

    # dcp.load(sd_out, storage_reader=storage_reader)

    end_load = perf_counter()

    if rank == 0:
        print(f"Time taken to load: {end_load - start_load} seconds")

    # dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--thread_count", type=int, default=1)
    parser.add_argument("--region", type=str, default="eu-north-1")
    parser.add_argument("--uri", type=str)
    args = parser.parse_args()

    backend = args.backend
    # dist.init_process_group(backend)
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    # print(f"Starting for rank {rank}, world_size is {world_size}")
    rank, world_size = 0, 1
    thread_count = args.thread_count

    region = args.region
    uri = args.uri
    suffix = "experiment"
    checkpoint_sharding_strategy = "hybrid"
    run_fsdp(rank, world_size, thread_count, backend, region, uri, suffix, checkpoint_sharding_strategy=checkpoint_sharding_strategy)
