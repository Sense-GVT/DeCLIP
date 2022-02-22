import torch
import numpy as np

from nvidia.dali.plugin.pytorch import to_torch_type, feed_ndarray
from .pipelines import _PipelineBase


# modified from linklink.dali
class DaliDataloader(object):
    """
    Arguments:
        pipeline (Pipeline):
            a :class:`linklink.dali.CustomPipeline` which will be running.
        batch_size (int):
            batch size used.
        epoch_size (int):
            the whole data size.
        num_threads (int, optional, default: 1):
            number of threads used to run the pipeline.
        device_id (int, optional, default: -1):
            device id dali used to run the pipeline, default to the default
            device PyTorch used.
        last_iter (int, optional, default: 0):
            last iteration this dataloader used, for recovery.
        verbose (bool, optional, default: False):
            whether to print debug information.
        dataset (Dataset): just for evaluation.

    """
    def __init__(self,
                 pipeline,
                 batch_size,
                 epoch_size,
                 num_threads=1,
                 device_id=-1,
                 last_iter=0,
                 verbose=False,
                 dataset=None):

        if device_id == -1:
            device_id = torch.cuda.current_device()

        self.dataset = dataset
        self.verbose = verbose
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.last_iter = last_iter
        if verbose:
            print('batch size: {}, epoch_size: {}, num_threads: {}, device: {}'
                  .format(batch_size, epoch_size, num_threads, device_id))
        self.pipe = _PipelineBase(pipeline, batch_size, num_threads, device_id,
                                  **pipeline.extra_kwargs)

    def __iter__(self):
        return _DataLoaderIter(self)

    def __len__(self):
        return (self.epoch_size + self.batch_size - 1) // self.batch_size

    def __del__(self):
        pass

    def reset_pipeline(self, pipeline):
        new_pipe = _PipelineBase(pipeline,
                                 self.pipe.batch_size,
                                 self.pipe.num_threads,
                                 self.pipe.device_id,
                                 **pipeline.extra_kwargs)
        self.pipe = new_pipe


class _DataLoaderIter(object):

    def __init__(self, loader):
        self.pipe = loader.pipe
        self.verbose = loader.verbose
        self.loader = loader

        self._size = len(loader)
        self.pipe.build()

        self._data_batches = [None, None]
        self._counter = loader.last_iter
        self._current_data_batch = 0
        self._last_batch_size = loader.epoch_size % loader.batch_size

        self._data_read_time = 0
        self.pipe.schedule_run()
        self._prefetch_counter = 0
        self._first_batch = None
        self._first_batch = self.next()

    def reset(self):
        self.pipe.reset()

    def _run_one_step(self, pipeline, data_batches, current_data_batch):
        p = pipeline
        outputs = p.share_outputs()
        device_id = p.device_id

        tensors = list()
        shapes = list()
        for out in outputs:
            tensors.append(out.as_tensor())
            shapes.append(tensors[-1].shape())

        if data_batches[current_data_batch] is None:
            torch_types = list()
            torch_devices = list()
            torch_gpu_device = torch.device('cuda', device_id)
            torch_cpu_device = torch.device('cpu')
            for i in range(len(outputs)):
                torch_types.append(to_torch_type[np.dtype(tensors[i].dtype())])
                from nvidia.dali.backend import TensorGPU
                if type(tensors[i]) is TensorGPU:
                    torch_devices.append(torch_gpu_device)
                else:
                    torch_devices.append(torch_cpu_device)

            pyt_tensors = list()
            for i in range(len(outputs)):
                pyt_tensors.append(torch.zeros(
                    shapes[i],
                    dtype=torch_types[i],
                    device=torch_devices[i]))

            data_batches[current_data_batch] = pyt_tensors
        else:
            pyt_tensors = data_batches[current_data_batch]

        for tensor, pyt_tensor in zip(tensors, pyt_tensors):
            feed_ndarray(tensor, pyt_tensor)

        p.release_outputs()
        p.schedule_run()

        return pyt_tensors

    def _check_stop(self):
        if self._counter >= self._size:
            self.reset()
            raise StopIteration
        else:
            self._counter += 1

        if self.pipe._last_iter:
            self._prefetch_counter += 1
            if self._prefetch_counter == self.pipe._prefetch_queue_depth:
                self.reset()
                raise StopIteration

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch

        self._check_stop()
        batch = self._run_one_step(self.pipe,
                                   self._data_batches,
                                   self._current_data_batch)

        self._current_data_batch = (self._current_data_batch + 1) % 2
        if self.verbose and self._counter % 10 == 0:
            pass

        # chop off last batch
        if self._counter >= self._size and self._last_batch_size > 0:
            if self.verbose:
                pass

            batch = [item[:self._last_batch_size] for item in batch]

        return {'image': batch[0], 'label': batch[1]}

    next = __next__

    def __iter__(self):
        return self
