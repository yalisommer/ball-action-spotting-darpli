import queue
from typing import Type
from multiprocessing import Queue

from rosny import ProcessStream, ComposeStream

from src.frame_fetchers import AbstractFrameFetcher, OpencvFrameFetcher
from src.data_loaders.base_data_loader import BaseDataLoader
from src.datasets import ActionDataset


class RandomSeekWorkerStream(ProcessStream):
    def __init__(self,
                 dataset: ActionDataset,
                 index_queue: Queue,
                 result_queue: Queue,
                 frame_fetcher_class: Type[AbstractFrameFetcher],
                 gpu_id: int = 0,
                 timeout: float = 1.0):
        super().__init__()
        self._dataset = dataset
        self._index_queue = index_queue
        self._result_queue = result_queue
        self._frame_fetcher_class = frame_fetcher_class
        self._gpu_id = gpu_id
        self._timeout = timeout

    def work(self):
        try:
            index = self._index_queue.get(timeout=self._timeout)
        except queue.Empty:
            return
        sample = self._dataset.get(index, self._frame_fetcher_class, self._gpu_id)
        self._result_queue.put(sample)


class RandomSeekWorkersStream(ComposeStream):
    def __init__(self, streams: list[RandomSeekWorkerStream]):
        self._streams = {f"worker_{i}": stream for i, stream in enumerate(streams)}
        super().__init__()


class RandomSeekDataLoader(BaseDataLoader):
    def __init__(self,
                 dataset: ActionDataset,
                 batch_size: int,
                 num_workers: int = 1,
                 gpu_id: int = 0):
        self.num_workers = num_workers
        super().__init__(dataset=dataset, batch_size=batch_size, gpu_id=gpu_id)

    def init_workers_stream(self) -> RandomSeekWorkersStream:
        streams = [
            RandomSeekWorkerStream(self.dataset, self._index_queue, self._result_queue,
                                 OpencvFrameFetcher, self.gpu_id)
            for _ in range(self.num_workers)
        ]
        return RandomSeekWorkersStream(streams)
