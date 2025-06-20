import torch
import numpy as np
import os
from colorama import Fore, Style, init as colorama_init
import inspect
import time
import atexit

colorama_init()


class Timer:
    def __init__(self):
        self.curr_state = 'init'
        self.curr_time = time.time()
        self.record = {}
        atexit.register(self.log)
        # colorama_init()

    def tick(self, next_state=None):
        if next_state is None:
            # 获取调用 tick 的上一帧
            frame = inspect.stack()[1]
            filepath = os.path.relpath(frame.filename)
            lineno = frame.lineno
            next_state = f"{filepath}:{lineno}"
        next_time = time.time()
        self._add_record(self.curr_state, next_state,
                         next_time - self.curr_time)
        self.curr_state = next_state
        self.curr_time = next_time

    def _add_record(self, s0, s1, delta):
        if (s0, s1) not in self.record:
            self.record[(s0, s1)] = []
        self.record[(s0, s1)].append(delta)

    def log(self):
        if self.curr_state == 'init':
            return
        # for key in self.record.keys():
        #     mean_delta = sum(self.record[key]) / len(self.record[key])
        #     print(f'{key}: {mean_delta}s, sum: {len(self.record[key])}')
        for (s0, s1), times in self.record.items():
            count = len(times)
            total = sum(times)
            avg = total / count
            print(
                f"{Fore.YELLOW}{s0} {Fore.GREEN}→ {Fore.YELLOW}{s1}: "
                f"{Fore.MAGENTA}avg {avg:.6f}s, "
                f"{Fore.BLUE}total {total:.6f}s, "
                f"{Fore.CYAN}count {count}"
            )


timer = Timer()


def _show(a):
    if isinstance(a, dict):
        result = {}
        for k in a.keys():
            result[k] = _show(a[k])
    elif isinstance(a, list):
        if len(a):
            result = f'list[{len(a)}]', _show(a[0])
        else:
            result = f'list[]'
    elif isinstance(a, tuple):
        result = (_show(x) for x in a)
    elif isinstance(a, np.ndarray):
        result = f'np.ndarray: {a.dtype}, {a.shape}'
    elif isinstance(a, torch.Tensor):
        result = f'torch.Tensor: {a.dtype}, {a.shape}, {a.device}'
    else:
        result = f'{type(a)}'
    return result


def show(a):
    result = _show(a)
    return str(result)
