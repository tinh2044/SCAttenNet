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
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
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
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", log_dir="logs", prefix="train"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = Logger(log_dir=log_dir, prefix=prefix)

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
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
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
                    if os.environ.get("LOCAL_RANK", "0") == "0":
                        log_str = self._format_log(
                            i, len(iterable), eta_string, iter_time, data_time, MB
                        )
                        self.logger.info(log_str)
                else:
                    log_str = self._format_log(
                        i, len(iterable), eta_string, iter_time, data_time, MB
                    )
                    self.logger.info(log_str)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            if os.environ.get("LOCAL_RANK", "0") == "0":
                final_str = "{} Total time: {} ({:.4f} s / it)".format(
                    header, total_time_str, total_time / len(iterable)
                )
                self.logger.info(final_str)
        else:
            final_str = "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
            self.logger.info(final_str)

    def _format_log(self, i, total_len, eta_string, iter_time, data_time, MB):
        if torch.cuda.is_available():
            return "Step [{}/{}]\tETA: {}\t{}\tTime: {}\tData: {}\tMax Memory: {:.0f}MB".format(
                i,
                total_len,
                eta_string,
                str(self),
                str(iter_time),
                str(data_time),
                torch.cuda.max_memory_allocated() / MB,
            )
        else:
            return "Step [{}/{}]\tETA: {}\t{}\tTime: {}\tData: {}".format(
                i, total_len, eta_string, str(self), str(iter_time), str(data_time)
            )


class Logger:
    def __init__(self, log_dir="logs", prefix="logfile"):
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = f"{log_dir}/{prefix}_{timestamp}.log"

        # Remove default handler
        logger.remove()

        # Add handlers for both console and file with custom format
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {message}",
            level="INFO",
            colorize=True,
        )
        logger.add(
            self.log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            level="INFO",
            rotation="10MB",
        )

        logger.info(f"Logging to {self.log_file}")

    def write(self, message):
        """Write log using loguru"""
        logger.info(message.strip())

    def flush(self):
        """Flush is handled by loguru"""
        pass

    @staticmethod
    def info(msg):
        """Log at INFO level"""
        logger.info(msg)

    @staticmethod
    def warning(msg):
        """Log at WARNING level"""
        logger.warning(msg)

    @staticmethod
    def error(msg):
        """Log at ERROR level"""
        logger.error(msg)
