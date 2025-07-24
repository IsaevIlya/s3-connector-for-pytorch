#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD
import contextlib
import io
import logging
import os
import time
import urllib.parse
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union, Optional, Tuple, Set, Callable
from typing import List

from s3torchconnectorclient._mountpoint_s3_client import S3Exception
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
    wait_random_exponential,
)
from torch.distributed.checkpoint.filesystem import (
    FileSystemReader,
    FileSystemWriter,
    FileSystemBase,
)
import torch

from s3torchconnector._s3client import S3Client
from s3torchconnector._s3dataset_common import parse_s3_uri
from ..s3reader import S3ReaderConstructor, S3ReaderConstructorProtocol
from .. import S3ClientConfig, S3Reader
from .s3_prefix_strategy import S3PrefixStrategyBase, DefaultPrefixStrategy
from .._user_agent import UserAgent

logger = logging.getLogger(__name__)


class S3FileSystem(FileSystemBase):
    def __init__(
        self,
        region: str,
        s3_client: Optional[S3Client] = None,
        s3client_config: Optional[S3ClientConfig] = None,
        reader_constructor: Optional[S3ReaderConstructorProtocol] = None,
    ) -> None:
        """
        Initialize S3FileSystem.

        Args:
            region (str): The AWS region for S3.
            s3_client (Optional[S3Client]): Optional S3Client instance.
            s3client_config (Optional[S3ClientConfig]): Optional S3ClientConfig with parameters for S3 client.
            reader_constructor (Optional[S3ReaderConstructorProtocol]): Optional partial(S3Reader) created using S3ReaderConstructor
                e.g. S3ReaderConstructor.sequential() or S3ReaderConstructor.range_based()
        """
        self._path: Union[str, os.PathLike] = ""
        self._reader_constructor = reader_constructor or S3ReaderConstructor.default()

        # Get reader type string for user agent
        reader_type_string = S3ReaderConstructor.get_reader_type_string(
            self._reader_constructor
        )
        user_agent = UserAgent(
            ["dcp", torch.__version__, f"md/reader_type#{reader_type_string}"]
        )

        self._client = (
            S3Client(
                region=region, user_agent=user_agent, s3client_config=s3client_config
            )
            if s3_client is None
            else s3_client
        )

    @contextmanager
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        """
        Create a stream for reading or writing to S3.

        Args:
            path (Union[str, os.PathLike]): The S3 path to read or write.
            mode (str): The mode for the stream. Supports 'rb' for read mode and 'wb' for write mode.

        Yields:
            io.BufferedIOBase: A stream for reading or writing to S3.

        Raises:
            ValueError: If the mode is not 'rb' or 'wb'.
        """
        path_str = _path_or_str_to_str(path)
        bucket, key = parse_s3_uri(path_str)

        if mode == "wb":  # write mode
            logger.debug("create_stream writable for %s", path_str)
            with self._client.put_object(bucket, key) as stream:
                yield stream
        elif mode == "rb":  # read mode
            logger.debug("create_stream readable for %s", path_str)
            with self._client.get_object(
                bucket, key, reader_constructor=self._reader_constructor
            ) as stream:
                yield stream
        else:
            raise ValueError(
                f"Invalid {mode=} mode argument: create_stream only supports rb (read mode) & wb (write mode)"
            )

    def concat_path(self, path: Union[str, os.PathLike], suffix: str) -> str:
        """
        Concatenate a suffix to the given path.

        Args:
            path (Union[str, os.PathLike]): The base path.
            suffix (str): The suffix to concatenate.

        Returns:
            str: The concatenated path.
        """
        logger.debug("concat paths %s and %s", path, suffix)
        path_str = os.fspath(path)
        result = os.path.join(path_str, suffix)
        return result

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        """
        Initialize the path for the filesystem.

        Args:
            path (Union[str, os.PathLike]): The path to initialize.

        Returns:
            Union[str, os.PathLike]: The initialized path.
        """
        logger.debug("init_path for %s", path)
        self._path = path
        return self._path

    def rename(
        self, old_path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        """Rename an object in S3.

        This is emulated by copying it to a new path and deleting the old path. The deletion part is retried (see also
        :func:`S3FileSystem._delete_with_retry`).

        Args:
            old_path (Union[str, os.PathLike]): The current path of the object.
            new_path (Union[str, os.PathLike]): The new path for the object.

        Raises:
            ValueError: If the old and new paths point to different buckets.
            S3Exception: If there is an error with the S3 client.
        """
        logger.debug("rename %s to %s", old_path, new_path)

        old_path_str = _path_or_str_to_str(old_path)
        new_path_str = _path_or_str_to_str(new_path)

        old_bucket, old_key = parse_s3_uri(old_path_str)
        escaped_old_key = self._escape_path(old_key)
        logger.debug("rename: escaped version of the source key: %s", escaped_old_key)
        new_bucket, new_key = parse_s3_uri(new_path_str)

        if old_bucket != new_bucket:
            raise ValueError(
                f"Source and destination buckets cannot be different (rename does not support cross-buckets operations)"
            )

        self._client.copy_object(
            src_bucket=old_bucket,
            src_key=escaped_old_key,
            dst_bucket=new_bucket,
            dst_key=new_key,
        )
        logger.debug("rename: copied %s to %s successfully", old_path_str, new_path_str)
        self._delete_with_retry(old_bucket, old_key)
        logger.debug("rename: s3://%s/%s successfully", old_bucket, old_key)

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        """No-op method for creating directories in S3 (not needed)."""
        pass

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        logger.debug("exists %s", path)

        path_str = _path_or_str_to_str(path)
        bucket, key = parse_s3_uri(path_str)
        try:
            self._client.head_object(bucket, key)
        except S3Exception as e:
            if str(e) != "Service error: The object was not found":
                raise
            return False
        return True

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        logger.debug("remove %s", path)

        path_str = _path_or_str_to_str(path)
        bucket, key = parse_s3_uri(path_str)
        try:
            self._client.delete_object(bucket, key)
        except S3Exception:
            logger.exception("Failed to remove object from S3")

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        logger.debug("validate_checkpoint_id for %s", checkpoint_id)

        if isinstance(checkpoint_id, Path):
            return True

        try:
            parse_s3_uri(_path_or_str_to_str(checkpoint_id))
        except ValueError:
            return False
        return True

    @retry(
        retry=retry_if_exception_type(S3Exception),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.ERROR),
        reraise=True,
    )
    def _delete_with_retry(self, bucket_name: str, old_key: str):
        """Wrapper around :func:`S3Client.delete_object` to retry the deletion.

        Will retry a maximum of 3 times, only for `S3Exception`s, and wait between retries. It will reraise the caught
        exception too, and logs retries and final error, if any."""
        self._client.delete_object(bucket_name, old_key)

    @staticmethod
    def _escape_path(string):
        """URL-encodes path segments while preserving '/' separators using urllib.parse.quote().

        Args:
            string (str): URL path string to escape

        Returns:
            str: Path string with each segment percent-encoded, separators preserved
        """
        if not string:
            return string
        parts = []
        for part in string.split("/"):
            parts.append(urllib.parse.quote(part, safe=""))
        return "/".join(parts)


from torch.distributed.checkpoint.planner import SavePlan, WriteItem
import dataclasses
from dataclasses import dataclass


@dataclass
class StorageMetadata:
    """Metadata for S3 storage prefix."""

    prefix: str


from typing import Callable
from torch.distributed.checkpoint.filesystem import _split_by_size_and_type as original_split
def _ordered_split_by_size_and_type(bins: int, items: List[WriteItem], sort_key: Optional[Callable] = None) -> List[List[WriteItem]]:
    buckets = original_split(bins, items)
    for bucket in buckets:
        if sort_key:
            bucket.sort(key=sort_key)
        else:
            bucket.sort(key=lambda item: item.index.fqn)
    print("****************REORDER TENSORS**************************")
    return buckets

class S3StorageWriter(FileSystemWriter):
    def __init__(
        self,
        region: str,
        path: str,
        s3client_config: Optional[S3ClientConfig] = None,
        prefix_strategy: Optional[S3PrefixStrategyBase] = None,
        sort_key: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        """
        Initialize an S3 writer for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (str): The S3 URI to write checkpoints to.
            s3client_config (Optional[S3ClientConfig]): Optional S3ClientConfig with parameters for S3 client.
            prefix_strategy (Optional[S3PrefixStrategyBase]): Optional strategy for generating S3 prefixes to
                optimize checkpoint organization and prevent throttling.
            kwargs (dict): Keyword arguments to pass to the parent :class:`FileSystemWriter`.
        """
        super().__init__(
            path=path,
            sync_files=False,  # FIXME: setting this to True makes the run to fail (L#333: `os.fsync(stream.fileno())`)
            **kwargs,
        )
        self.fs = S3FileSystem(region, s3client_config=s3client_config)  # type: ignore
        self.path = self.fs.init_path(path)
        self.prefix_strategy = prefix_strategy or DefaultPrefixStrategy()
        self.sort_key = sort_key

        # Replace the original split function with our custom one
        import torch.distributed.checkpoint.filesystem as fs_module
        # Use functools.partial to bind the sort_key parameter
        from functools import partial
        fs_module._split_by_size_and_type = partial(_ordered_split_by_size_and_type, sort_key=sort_key)

    def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
        """
        Prepare save plans with S3-specific storage metadata.

        Args:
            plans: List of save plans to be processed.

        Returns:
            Modified save plans with S3 storage metadata.
        """
        return [
            dataclasses.replace(
                plan, storage_data=StorageMetadata(self.prefix_strategy(idx))
            )
            for idx, plan in enumerate(plans)
        ]

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)

from torch.futures import Future
from typing import Dict, cast, IO
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, ReadItem, LoadItemType
from torch import Tensor
from torch.distributed._shard._utils import narrow_tensor_by_index
import heapq
import concurrent.futures
from dataclasses import dataclass
from queue import Queue
from threading import Lock

class ByteArrayIO:
    def __init__(self, buf: bytearray):
        self.buf = buf
        self.pos = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.buf) - self.pos
        end = min(self.pos + size, len(self.buf))
        start = self.pos
        self.pos = end
        return  self.buf[start:end]

    def readinto(self, b: bytearray) -> int:
        n = len(b)
        remaining = len(self.buf) - self.pos
        if remaining < n:
            n = remaining
        b[:n] = self.buf[self.pos:self.pos + n]
        self.pos += n
        return n

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:  # SEEK_SET
            self.pos = offset
        elif whence == 1:  # SEEK_CUR
            self.pos += offset
        elif whence == 2:  # SEEK_END
            self.pos = len(self.buf) + offset
        return self.pos

    def tell(self) -> int:
        return self.pos


class _ReaderView(io.IOBase):
    def __init__(self, base_stream: S3Reader, offset: int, len: int):
        super().__init__()
        self.offset = offset
        self.len = len
        self.base_stream = base_stream
        self.seek(0)

    def seek(self, offset: int, whence: int = os.SEEK_SET, /) -> int:
        if whence == os.SEEK_SET:
            offset = self.offset + offset
        elif whence == os.SEEK_END:
            whence = os.SEEK_SET
            offset = (self.offset + self.len) - offset
        return self.base_stream.seek(offset, whence)

    def tell(self) -> int:
        return self.base_stream.tell() - self.offset

    def readable(self) -> bool:
        return self.base_stream.readable()

    def seekable(self) -> bool:
        return self.base_stream.seekable()

    def readinto(self, b):
        max_size = self.len - self.tell()
        if max_size == 0:
            return 0
        if len(b) > max_size:
            b = memoryview(b)[:max_size]
        return self.base_stream.readinto(b)  # type: ignore[attr-defined]

    def read(self, size=-1):
        max_size = self.len - self.tell()
        if size == -1 or size > max_size:
            size = max_size
        return self.base_stream.read(size)

    def read1(self, size=-1):
        max_size = self.len - self.tell()
        if size == -1 or size > max_size:
            size = max_size
        return self.base_stream.read1(size)

    def prefetch(self, size):
        self.base_stream.prefetch(size)


@dataclass
class BucketInfo:
    items: List[ReadItem]
    start_offset: int
    end_offset: int
    total_size: int

class S3StorageReader(FileSystemReader):
    def __init__(
        self,
        region: str,
        path: Union[str, os.PathLike],
        s3client_config: Optional[S3ClientConfig] = None,
        reader_constructor: Optional[S3ReaderConstructorProtocol] = None,
        use_custom_load: bool = False
    ) -> None:
        """
        Initialize an S3 reader for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (Union[str, os.PathLike]): The S3 path to read checkpoints from.
            s3client_config (Optional[S3ClientConfig]): Optional S3ClientConfig with parameters for S3 client.
            reader_constructor (Optional[S3ReaderConstructorProtocol]): Optional partial(S3Reader) created using S3ReaderConstructor
                e.g. S3ReaderConstructor.sequential() or S3ReaderConstructor.range_based()
        """
        super().__init__(path)
        self.fs = S3FileSystem(region, s3client_config=s3client_config, reader_constructor=reader_constructor)  # type: ignore
        self.path = self.fs.init_path(path)
        self.sync_files = False
        self.GAP_THRESHOLD = 1024*1024
        self.use_custom_load = use_custom_load

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)

    def _create_buckets(self, heap_items: List[Tuple[int, ReadItem]], num_buckets: int) -> List[BucketInfo]:
        buckets: List[BucketInfo] = []
        current_items: List[Tuple[int, ReadItem]] = []
        current_size = 0
        last_end = None
        total_size = sum(self.storage_data[item[1].storage_index].length for item in heap_items)
        target_bucket_size = total_size / num_buckets

        while heap_items:
            offset, item = heapq.heappop(heap_items)
            item_md = self.storage_data[item.storage_index]
            item_size = item_md.length

            print(f"dest {item.dest_index.fqn} \t\t storage {item.storage_index.fqn} \t\t offset {offset}")

            # Check if there's a large gap
            if current_size >= target_bucket_size or last_end is not None and (item_md.offset - last_end) > self.GAP_THRESHOLD:
            # if last_end is not None and (
            #         item_md.offset - last_end) > self.GAP_THRESHOLD:
                if current_items:
                    print(f"creating a new bucket, gap size is {item_md.offset - last_end}")
                    # Create a bucket for items before the gap
                    first_item_md = self.storage_data[current_items[0].storage_index]
                    last_item_md = self.storage_data[current_items[-1].storage_index]
                    buckets.append(BucketInfo(
                        items=current_items,
                        start_offset=first_item_md.offset,
                        end_offset=last_item_md.offset + last_item_md.length,
                        total_size=current_size
                    ))
                    current_items = []
                    current_size = 0

            current_items.append(item)
            current_size += item_size
            last_end = item_md.offset + item_md.length

        # Handle remaining items
        if current_items:
            first_item_md = self.storage_data[current_items[0].storage_index]
            last_item_md = self.storage_data[current_items[-1].storage_index]
            buckets.append(BucketInfo(
                items=current_items,
                start_offset=first_item_md.offset,
                end_offset=last_item_md.offset + last_item_md.length,
                total_size=current_size
            ))

        print(f"creating new buckets #{len(buckets)}")
        return buckets

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        if not self.use_custom_load:
            print("Using default load strategy.-----------------------------------")
            return super().read_data(plan, planner)

        print("Using CUSTOM load strategy.-----------------------------------")
        print(self.transforms)
        NUM_PARALLEL_STREAMS = 1

        per_file: Dict[str, List[ReadItem]] = dict()
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            if path not in per_file:
                per_file[path] = []
            heapq.heappush(per_file[path], (item_md.offset, read_item))

        for relative_path, reqs in per_file.items():
            new_path = self.fs.concat_path(self.path, relative_path)
            # Split requests into buckets
            buckets = self._create_buckets(reqs, NUM_PARALLEL_STREAMS)

            # Create a queue of buckets to process
            bucket_queue = Queue()
            for bucket in buckets:
                bucket_queue.put(bucket)

            # Keep track of active futures
            active_futures: Set[Future] = set()
            futures_lock = Lock()
            all_done = False

            def process_bucket(bucket: BucketInfo):
                with self.fs.create_stream(new_path, "rb") as stream:
                    stream.seek(bucket.start_offset)
                    length = bucket.end_offset - bucket.start_offset
                    try:
                        stream.prefetch(length)
                    except Exception as e:
                        print(f"Exception: {e}")

                    pref_offset = bucket.start_offset
                    print(f"bucket offset {pref_offset}, bucket length {length}, buckets coutn {len(bucket.items)}")
                    for req in bucket.items:
                        item_md = self.storage_data[req.storage_index]
                        assert (item_md.offset >= pref_offset), f"offset is not ordered {item_md.offset} is before {pref_offset}"
                        pref_offset = item_md.offset
                        assert (item_md.offset >= bucket.start_offset), f"item {item_md.offset} is before {bucket.start_offset}"
                        assert (item_md.offset + item_md.length <= bucket.end_offset), f"item end {item_md.offset + item_md.length} is after {bucket.end_offset}"

                        file_slice = _ReaderView(stream, item_md.offset, item_md.length)

                        transform_from = self.transforms.transform_load_stream(
                            req,
                            # This field wasn't present in older
                            # implementations so provide a fallback.
                            item_md.transform_descriptors or (),
                            file_slice,
                        )
                        if req.type == LoadItemType.BYTE_IO:
                            # read_bytes = file_slice.read1(item_md.length)
                            read_bytes = io.BytesIO(transform_from.read(-1))
                            read_bytes.seek(0)
                            planner.load_bytes(req, read_bytes)
                        else:
                            # tensor = cast(
                            #     Tensor,
                            #     torch.load(
                            #         cast(IO[bytes], file_slice.read1(item_md.length)),
                            #         map_location="cpu",
                            #         weights_only=True,
                            #     ),
                            # )

                            if transform_from.seekable():
                                seekable = transform_from
                            else:
                                # torch.load requires a seekable input, so read the transform
                                # stream now and store the output if needed
                                seekable = io.BytesIO(transform_from.read(-1))
                                seekable.seek(0)

                            try:
                                tensor = cast(
                                    Tensor,
                                    torch.load(
                                        seekable,
                                        map_location="cpu",
                                        weights_only=True,
                                    ),
                                )
                            except Exception as e:
                                print(f"Exception: {e}")
                                return


                            tensor = narrow_tensor_by_index(
                                tensor, req.storage_offsets, req.lengths
                            )
                            target_tensor = planner.resolve_tensor(req).detach()

                            assert (
                                    target_tensor.size() == tensor.size()
                            ), f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                            target_tensor.copy_(tensor)
                            planner.commit_tensor(req, target_tensor)

            def future_done_callback(future: Future):
                with futures_lock:
                    if not all_done:
                        try:
                            # Get next bucket if available
                            next_bucket = bucket_queue.get_nowait()
                            # Submit new task
                            new_future = executor.submit(process_bucket, next_bucket)
                            active_futures.add(new_future)
                            new_future.add_done_callback(future_done_callback)
                        except Exception:
                            # No more buckets to
                            pass

                    # Remove the completed future
                    active_futures.remove(future)

            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PARALLEL_STREAMS) as executor:
                # Initial submission of tasks up to NUM_PARALLEL_STREAMS
                for _ in range(min(NUM_PARALLEL_STREAMS, len(buckets))):
                    bucket = bucket_queue.get()
                    future = executor.submit(process_bucket, bucket)
                    with futures_lock:
                        active_futures.add(future)
                    future.add_done_callback(future_done_callback)

                # Wait for all tasks to complete
                while True:
                    with futures_lock:
                        if not active_futures and bucket_queue.empty():
                            break
                    time.sleep(0.001)  # Small sleep to prevent busy waiting

                # Mark as done to prevent new submissions
                all_done = True

                # Check for any exceptions
                for future in concurrent.futures.as_completed(active_futures):
                    future.result()  # Will raise any exceptions that occurred

        fut: Future = Future()
        fut.set_result(None)
        return fut




def _path_or_str_to_str(path: Union[str, os.PathLike]) -> str:
    return path if isinstance(path, str) else str(path)
