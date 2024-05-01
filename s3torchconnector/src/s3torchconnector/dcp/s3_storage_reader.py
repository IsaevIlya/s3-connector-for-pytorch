#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
from typing import List

from torch.distributed.checkpoint import StorageReader, Metadata, LoadPlan, LoadPlanner
from torch.futures import Future


class S3StorageReader(StorageReader):
    def __init__(self):
        pass

    def read_metadata(self) -> Metadata:
        pass

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        pass

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        pass

    def prepare_global_plan(self, plans: List[LoadPlan]) -> List[LoadPlan]:
        pass

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        pass