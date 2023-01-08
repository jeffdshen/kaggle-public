import collections
from types import SimpleNamespace
import numpy as np
import numbers
import os
import torch
from torch.optim.lr_scheduler import LambdaLR

import random
import copy


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = collections.deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Allows repeats, but also works for n < batch_size
        samples = random.choices(self.buffer, k=batch_size)
        batch = tuple(stack(col) for col in zip(*samples))
        return batch


def to_torch(x, device=None):
    if isinstance(x, dict):
        return {k: to_torch(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return tuple(to_torch(v, device) for v in x)
    if isinstance(x, np.ndarray):
        y = torch.from_numpy(x)
        return y if device is None else y.to(device)
    raise ValueError(f"Unrecognized type: {type(x)}")


def stack(xs):
    if len(xs) == 0:
        return xs
    x = xs[0]
    if isinstance(x, dict):
        return {k: stack([y[k] for y in xs]) for k in x}
    if isinstance(x, np.ndarray):
        return np.array(xs)
    return np.array(xs)
    # raise ValueError(f"Unrecognized batch type: {type(x)}")


def select(x, i):
    if isinstance(x, dict):
        return {k: select(v, i) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x[i]
    if isinstance(x, (list, tuple)):
        return x[i]
    raise ValueError(f"Unrecognized batch type: {type(x)}")


def masked_select(x, mask):
    if isinstance(x, dict):
        return {k: masked_select(v, mask) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x[mask]
    if isinstance(x, (list, tuple)):
        return [xx for xx, m in zip(x, mask) if m]
    raise ValueError(f"Unrecognized batch type: {type(x)}")


def make_transitions(prev_obs, obs, actions, rewards, terminated, infos, num_envs):
    final_obs = infos.get("final_observation", [0] * num_envs)
    final_obs_done = infos.get("_final_observation", [False] * num_envs)
    transitions = []
    for i in range(num_envs):
        state = select(prev_obs, i)
        action = select(actions, i)
        reward = select(rewards, i)
        next_state = (
            select(final_obs, i) if select(final_obs_done, i) else select(obs, i)
        )
        done = select(terminated, i)
        transitions.append((state, action, reward, next_state, done))
    transitions = copy.deepcopy(transitions)
    return transitions


# Compute the max of last num_dims dims
# Returns a torch tensor
def multi_max(x, num_dims):
    x = x.view(*x.size()[:-num_dims], -1)
    return x.amax(dim=-1)


# Compute argmax of last num_dims dims
# Returns a numpy array
def multi_argmax(x, num_dims):
    size = x.size()[-num_dims:]
    x = x.view(*x.size()[:-num_dims], -1)
    a = x.argmax(dim=-1).cpu().detach().numpy()

    return np.stack(np.unravel_index(a, size), axis=-1)


def get_linear_warmup_power_decay_scheduler(
    optimizer, warmup_steps, max_num_steps, end_multiplier=0.0, power=1
):
    """Uses a power function a * x^power + b, such that it equals 1.0 at start_step=1
    and the end_multiplier at end_step. Afterwards, returns the end_multiplier forever.
    For the first warmup_steps, linearly increase the learning rate until it hits the power
    learning rate.
    """

    # a = end_lr - start_lr / (end_step ** power - start_step ** power)
    start_multiplier = 1.0
    start_step = 1
    scale = (end_multiplier - start_multiplier) / (
        max_num_steps**power - start_step**power
    )
    # b = start_lr - scale * start_step ** power
    constant = start_multiplier - scale * (start_step**power)

    def lr_lambda(step):
        step = start_step + step
        if step < warmup_steps:
            warmup_multiplier = scale * (warmup_steps**power) + constant
            return float(step) / float(max(1, warmup_steps)) * warmup_multiplier
        elif step >= max_num_steps:
            return end_multiplier
        else:
            return scale * (step**power) + constant

    return LambdaLR(optimizer, lr_lambda)


class NoopWandB:
    def __init__(self):
        self.run = SimpleNamespace(summary={}, name="peach-moon-1")
        self.config = {}

    def log(self, *_args, **_kwargs):
        pass

    def save(self, *_args, **_kwargs):
        pass


def result_to_str(result):
    if isinstance(result, np.ndarray):
        result = result.item()
    if isinstance(result, int):
        return f"{result:8d}"
    elif isinstance(result, float):
        return f"{result:10.5}"
    else:
        raise ValueError(f"Unsupported result of type {type(result)}: {result}")


def results_to_str(results):
    results_str = [f"{key}: {result_to_str(result)}" for key, result in results.items()]
    results_str = " | ".join(results_str)
    return results_str


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.use_deterministic_algorithms(False)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.use_deterministic_algorithms(True)
