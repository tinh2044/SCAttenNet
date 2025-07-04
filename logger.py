import os
import sys
import time
import datetime
from collections import defaultdict, deque

import torch
import torch.distributed as dist
from loguru import logger

from utils import is_dist_avail_and_initialized

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)



class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    if os.environ['LOCAL_RANK'] == '0':
                        if torch.cuda.is_available():
                            print(log_msg.format(
                                i, len(iterable), eta=eta_string,
                                meters=str(self),
                                time=str(iter_time), data=str(data_time),
                                memory=torch.cuda.max_memory_allocated() / MB))
                        else:
                            print(log_msg.format(
                                i, len(iterable), eta=eta_string,
                                meters=str(self),
                                time=str(iter_time), data=str(data_time)))
                else:
                    if torch.cuda.is_available():
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB))
                    else:
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            if os.environ['LOCAL_RANK'] == '0':
                print('{} Total time: {} ({:.4f} s / it)'.format(
                    header, total_time_str, total_time / len(iterable)))
        else:
            print('{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)))
            

class Logger:
    def __init__(self, log_dir="logs", prefix="logfile"):
        os.makedirs(log_dir, exist_ok=True)  # Tạo thư mục nếu chưa có

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = f"{log_dir}/{prefix}_{timestamp}.log"

        logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")  # Log ra console
        logger.add(self.log_file, format="{time} {level} {message}", level="INFO", rotation="10MB")  # Log ra file

        print(f"Logging to {self.log_file}")
        
    def write(self, message):
        """Ghi log thay thế print(), hỗ trợ live writing"""
        logger.info(message.strip())
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message.strip() + "\n")  # Ghi vào file ngay lập tức
            f.flush()  # Đảm bảo ghi ngay

        # sys.__stdout__.write(message)  # Ghi ra console ngay lập tức

    def flush(self):
        """Flush dữ liệu (không cần thiết do đã gọi flush() trong write)"""
        pass  

    @staticmethod
    def info(msg):
        """Ghi log mức INFO"""
        logger.info(msg)

    @staticmethod
    def warning(msg):
        """Ghi log mức WARNING"""
        logger.warning(msg)

    @staticmethod
    def error(msg):
        """Ghi log mức ERROR"""
        logger.error(msg)

