import os

import numpy as np
import torch
from tqdm import tqdm
import tensorflow as tf
from itertools import groupby

import torch.distributed as dist

total_body_idx = 33
total_hand = 42

body_idx = list(range(11, 17))
lefthand_idx = [x + total_body_idx for x in range(0, 21)]

righthand_idx = [x + 21 for x in lefthand_idx]

total_idx = body_idx + lefthand_idx + righthand_idx


def save_checkpoints(model, optimizer, path_dir, epoch, name=None):
    if not os.path.exists(path_dir):
        print(f"Making directory {path_dir}")
        os.makedirs(path_dir)
    if name is None:
        filename = f"{path_dir}/checkpoints_{epoch}.pth"
    else:
        filename = f"{path_dir}/checkpoints_{epoch}_{name}.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        filename,
    )


def load_checkpoints(model, optimizer, path, resume=True):
    if not os.path.exists(path):
        raise FileNotFoundError
    if os.path.isdir(path):
        epoch = max([int(x[x.index("_") + 1 : len(x) - 4]) for x in os.listdir(path)])
        filename = f"{path}/checkpoints_{epoch}.pth"
        print(f"Loaded latest checkpoint: {epoch}")

        checkpoints = torch.load(filename)

    else:
        print(f"Load checkpoint from file : {path}")
        checkpoints = torch.load(path)

    model.load_state_dict(checkpoints["model"])
    optimizer.load_state_dict(checkpoints["optimizer"])
    if resume:
        return checkpoints["epoch"] + 1
    else:
        return 1


def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def ctc_decode(gloss_logits, beam_size, input_lengths):
    gloss_logits = gloss_logits.permute(1, 0, 2)
    gloss_logits = gloss_logits.cpu().detach().numpy()
    tf_gloss_logits = np.concatenate(
        (gloss_logits[:, :, 1:], gloss_logits[:, :, 0, None]),
        axis=-1,
    )
    ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
        inputs=tf_gloss_logits,
        sequence_length=input_lengths.cpu().detach().numpy(),
        beam_width=beam_size,
        top_paths=1,
    )
    ctc_decode = ctc_decode[0]
    tmp_gloss_sequences = [[] for i in range(input_lengths.shape[0])]
    for value_idx, dense_idx in enumerate(ctc_decode.indices):
        tmp_gloss_sequences[dense_idx[0]].append(
            ctc_decode.values[value_idx].numpy() + 1
        )
    decoded_gloss_sequences = []
    for seq_idx in range(0, len(tmp_gloss_sequences)):
        decoded_gloss_sequences.append(
            [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
        )
    return decoded_gloss_sequences


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    print(os.environ)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return
    args.distributed = True

    torch.cuda.set_device(args.gpu)

    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def expand_frame_mask(mask, num_heads, num_keypoints, num_queries):
    mask = mask[:, None, :, None, None]

    mask = mask.expand(-1, num_heads, -1, num_keypoints, num_queries)

    return mask


def check_state_dict(model, state_dict):
    is_same = True
    model_state_dict = model.state_dict()
    for key in state_dict.keys():
        if key not in model_state_dict:
            print(f"Key {key} not found in model state dict")
        else:
            if state_dict[key].shape != model_state_dict[key].shape:
                print("Has different shape in model state dict")
                print(f"Key: {key}")
                print(f"State dict shape: {state_dict[key].shape}")
                print(f"Model state dict shape: {model_state_dict[key].shape}")
                print()
                is_same = False
    if is_same:
        print("Model and state dict are the same")
    else:
        print("Model and state dict are different")
    return is_same
