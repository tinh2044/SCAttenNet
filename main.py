import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import time
import argparse
import json
import datetime
import numpy as np
import yaml
import random
from pathlib import Path
from loguru import logger

import torch.distributed as dist


from optimizer import build_optimizer, build_scheduler
from Tokenizer import GlossTokenizer
from dataset import SLR_Dataset
from model import MSCA_Net

from opt import train_one_epoch, evaluate_fn
import utils


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Visual-Language-Pretraining (VLP) V2 scripts", add_help=False
    )
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=100, type=int)

    parser.add_argument("--finetune", default="", help="finetune from checkpoint")

    parser.add_argument(
        "--world_size", default=2, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument(
        "--device", default="cpu", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--test_on_last_epoch",
        default=False,
        type=bool,
        help="Perform evaluation on last epoch",
    )
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument(
        "--cfg_path", type=str, required=True, help="Path to config file"
    )

    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    return parser


def main(args, cfg):
    model_dir = cfg["training"]["model_dir"]
    log_dir = f"{model_dir}/log"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    utils.init_distributed_mode(args)

    seed = args.seed + utils.get_rank()
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    cfg_data = cfg["data"]
    gloss_tokenizer = GlossTokenizer(config["gloss_tokenizer"])
    train_data = SLR_Dataset(
        root=cfg_data["root"],
        gloss_tokenizer=gloss_tokenizer,
        cfg=cfg_data,
        split="train",
    )
    dev_data = SLR_Dataset(
        root=cfg_data["root"],
        gloss_tokenizer=gloss_tokenizer,
        cfg=cfg_data,
        split="dev",
    )

    test_data = SLR_Dataset(
        root=cfg_data["root"],
        gloss_tokenizer=gloss_tokenizer,
        cfg=cfg_data,
        split="test",
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.data_collator,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    dev_dataloader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dev_data.data_collator,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=test_data.data_collator,
        pin_memory=True,
    )

    model = MSCA_Net(cfg=cfg["model"], gloss_tokenizer=gloss_tokenizer, device=device)
    model = model.to(device)
    n_parameters = utils.count_model_parameters(model)

    print(f"Number of parameters: {n_parameters}")

    # Calculate FLOPs
    # Calculate total number of joints (maximum index + 1)
    all_joint_indices = (
        cfg["model"]["body_idx"] + cfg["model"]["left_idx"] + cfg["model"]["right_idx"]
    )
    max_joint_index = max(all_joint_indices)

    # Add joint_parts if they exist in data config
    if "joint_parts" in cfg["data"]:
        for part in cfg["data"]["joint_parts"]:
            if isinstance(part, list):
                max_joint_index = max(max_joint_index, max(part))

    input_shape = {
        "batch_size": args.batch_size,
        "seq_len": cfg["data"]["max_len"],
        "num_joints": max_joint_index + 1,  # +1 because indices are 0-based
        "vocab_size": len(gloss_tokenizer),
    }

    print("Calculating FLOPs...")
    model_info = utils.get_model_info(model, input_shape, device)

    print("Model Information:")
    print(f"  Total parameters: {model_info['total_params']:,}")
    print(f"  Trainable parameters: {model_info['trainable_params']:,}")
    print(f"  Non-trainable parameters: {model_info['non_trainable_params']:,}")

    if "flops" in model_info:
        print(f"  FLOPs: {model_info['flops_str']}")
        print(f"  MACs: {model_info['macs_str']}")
        print(f"  Parameters (from thop): {model_info['params_str']}")
    print()

    if args.finetune:
        print(f"Finetuning from {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location="cpu")
        ret = model.load_state_dict(checkpoint["model"], strict=False)
        print("Missing keys: \n", "\n".join(ret.missing_keys))
        print("Unexpected keys: \n", "\n".join(ret.unexpected_keys))

    optimizer = build_optimizer(config=cfg["training"]["optimization"], model=model)
    # Update config with total epochs for warmup scheduler
    cfg["training"]["optimization"]["total_epochs"] = args.epochs
    scheduler, scheduler_type = build_scheduler(
        config=cfg["training"]["optimization"], optimizer=optimizer
    )
    output_dir = Path(cfg["training"]["model_dir"])

    if args.resume:
        print(f"Resume training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        if utils.check_state_dict(model, checkpoint["model"]):
            ret = model.load_state_dict(checkpoint["model"], strict=True)
        else:
            print("Model and state dict are different")
            raise ValueError("Model and state dict are different")

        if (
            not args.eval
            and "optimizer" in checkpoint
            and "scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            print("Loading optimizer and scheduler")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])

            print(f"New learning rate : {scheduler.get_last_lr()[0]}")
        args.start_epoch = checkpoint["epoch"] + 1

        print("Missing keys: \n", "\n".join(ret.missing_keys))
        print("Unexpected keys: \n", "\n".join(ret.unexpected_keys))

    if args.eval:
        if not args.resume:
            logger.warning(
                "Please specify the trained model: --resume /path/to/best_checkpoint.pth"
            )

        dev_results = evaluate_fn(
            args,
            dev_dataloader,
            model,
            epoch=0,
            beam_size=5,
            print_freq=args.print_freq,
            results_path=f"{model_dir}/dev_results.json",
            tokenizer=gloss_tokenizer,
            log_dir=f"{log_dir}/eval/dev",
        )
        print(
            f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_results['loss']:.3f}"
        )
        print(f"* DEV wer {dev_results['wer']:.3f}")

        test_results = evaluate_fn(
            args,
            test_dataloader,
            model,
            epoch=0,
            beam_size=5,
            print_freq=args.print_freq,
            results_path=f"{model_dir}/test_results.json",
            tokenizer=gloss_tokenizer,
            log_dir=f"{log_dir}/eval/test",
        )
        print(
            f"Test loss of the network on the {len(test_dataloader)} test videos: {test_results['loss']:.3f}"
        )
        print(f"* TEST wer {test_results['wer']:.3f}")
        return

    print(f"Training on {device}")
    print(
        f"Start training for {args.epochs} epochs and start epoch: {args.start_epoch}"
    )
    start_time = time.time()
    min_wer = 200
    for epoch in range(args.start_epoch, args.epochs):
        train_results = train_one_epoch(
            args,
            model,
            train_dataloader,
            optimizer,
            epoch,
            print_freq=args.print_freq,
            log_dir=f"{log_dir}/train",
        )
        scheduler.step()
        checkpoint_paths = [output_dir / f"checkpoint_{epoch}.pth"]
        prev_chkpt = output_dir / f"checkpoint_{epoch - 1}.pth"
        if os.path.exists(prev_chkpt):
            os.remove(prev_chkpt)
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        print()
        test_results = evaluate_fn(
            args,
            test_dataloader,
            model,
            epoch,
            beam_size=5,
            print_freq=args.print_freq,
            tokenizer=gloss_tokenizer,
            log_dir=f"{log_dir}/test",
        )
        dev_results = evaluate_fn(
            args,
            dev_dataloader,
            model,
            epoch,
            beam_size=5,
            print_freq=args.print_freq,
            tokenizer=gloss_tokenizer,
            log_dir=f"{log_dir}/dev",
        )

        if min_wer > test_results["wer"] or min_wer > dev_results["wer"]:
            min_wer = min(test_results["wer"], dev_results["wer"])
            checkpoint_paths = [output_dir / "best_checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )
        print(f"* DEV wer {test_results['wer']:.3f} Min DEV WER {min_wer}")

        log_results = {
            **{f"train_{k}": v for k, v in train_results.items()},
            **{f"test_{k}": v for k, v in test_results.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }
        print()
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_results) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def init_ddp(local_rank):
    torch.cuda.set_device(local_rank)
    os.environ["RANK"] = str(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser("MSCA scripts", parents=[get_args_parser()])
    args = parser.parse_args()
    with open(args.cfg_path, "r+", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # config.update({k: v for k, v in vars(args).items() if v is not None})
    Path(config["training"]["model_dir"]).mkdir(parents=True, exist_ok=True)
    main(args, config)
