import os
import torch
import pickle
import numpy as np
import linklink as link


def link_dist(func):

    def wrapper(*args, **kwargs):
        dist_init()
        func(*args, **kwargs)
        dist_finalize()

    return wrapper


def dist_init(method='slurm', device_id=0):
    if method == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        # ntasks = int(os.environ['SLURM_NTASKS'])
        # node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)
    elif method == 'single_node':
        torch.cuda.set_device(device_id)

    link.initialize()
    world_size = link.get_world_size()
    rank = link.get_rank()

    return rank, world_size


def dist_finalize():
    link.finalize()


def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(link.new_group(rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank//group_size]


class DistModule(torch.nn.Module):
    def __init__(self, module, sync=False):
        super(DistModule, self).__init__()
        self.module = module
        self.broadcast_params()

        self.sync = sync
        if not sync:
            self._grad_accs = []
            self._register_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _register_hooks(self):
        for i, (name, p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(name, p, i))
                self._grad_accs.append(grad_acc)

    def _make_hook(self, name, p, i):
        def hook(*ignore):
            link.allreduce_async(p.grad.data)
        return hook

    def sync_gradients(self):
        """ average gradients """
        if self.sync and link.get_world_size() > 1:
            for name, param in self.module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    link.allreduce(param.grad.data)
        else:
            link.synchronize()

    def broadcast_params(self):
        """ broadcast model parameters """
        for name, param in self.module.state_dict().items():
            link.broadcast(param, 0)


def _serialize_to_tensor(data, group=None):
    # backend = link.get_backend(group)
    # assert backend in ["gloo", "nccl"]
    # device = torch.device("cpu" if backend == "gloo" else "cuda")
    device = torch.cuda.current_device()

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        import logging
        logger = logging.getLogger('global')
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                link.get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def broadcast_object(obj, group=None):
    """make suare obj is picklable
    """
    if link.get_world_size() == 1:
        return obj

    serialized_tensor = _serialize_to_tensor(obj).cuda()
    numel = torch.IntTensor([serialized_tensor.numel()]).cuda()
    link.broadcast(numel, 0)
    # serialized_tensor from storage is not resizable
    serialized_tensor = serialized_tensor.clone()
    serialized_tensor.resize_(numel)
    link.broadcast(serialized_tensor, 0)
    serialized_bytes = serialized_tensor.cpu().numpy().tobytes()
    deserialized_obj = pickle.loads(serialized_bytes)
    return deserialized_obj
