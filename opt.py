import torch
import json
from collections import defaultdict
import datetime
import math
import sys
from metrics import wer_list

from logger import MetricLogger, SmoothedValue
from utils import ctc_decode


def train_one_epoch(
    args, model, data_loader, optimizer, epoch, print_freq=1, log_dir="log/train"
):
    model.train()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metric_logger = MetricLogger(
        delimiter="  ",
        log_dir=log_dir,
        file_name=f"epoch_{epoch}_({timestamp}).log",
    )
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Training epoch: [{epoch}/{args.epochs}]"
    for step, (src_input) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        optimizer.zero_grad()
        output = model(src_input)
        loss = output["total_loss"]
        with torch.autograd.set_detect_anomaly(True):
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                print("NaN loss")
        model.zero_grad()
        for k, v in output.items():
            if "loss" in k and "gloss" not in k:
                # print(k, v)
                metric_logger.update(**{k: v})
        # metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged results:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_fn(
    args,
    dataloader,
    model,
    epoch,
    beam_size=1,
    print_freq=1,
    results_path=None,
    tokenizer=None,
    log_dir="log/test",
):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model.eval()
    metric_logger = MetricLogger(
        log_dir=log_dir, file_name=f"epoch_{epoch}_({timestamp}).log"
    )
    header = f"Test epoch: [{epoch}/{args.epochs}]"
    print_freq = 10
    results = defaultdict(dict)

    with torch.no_grad():
        for _, (src_input) in enumerate(
            metric_logger.log_every(dataloader, print_freq, header)
        ):
            output = model(src_input)

            for k, gls_logits in output.items():
                if "_loss" in k and "gloss" not in k:
                    metric_logger.update(**{k: gls_logits})
                    continue
                if "gloss_logits" not in k:
                    continue
                logits_name = k.replace("gloss_logits", "")

                ctc_decode_output = ctc_decode(
                    gloss_logits=gls_logits,
                    beam_size=beam_size,
                    input_lengths=output["input_lengths"],
                )
                batch_pred_gls = tokenizer.batch_decode(ctc_decode_output)
                lower_case = tokenizer.lower_case
                for name, gls_hyp, gls_ref in zip(
                    src_input["name"], batch_pred_gls, src_input["gloss_input"]
                ):
                    results[name][f"{logits_name}_gls_hyp"] = (
                        gls_hyp.upper() if lower_case else gls_hyp
                    )
                    results[name]["gls_ref"] = (
                        gls_ref.upper() if lower_case else gls_ref
                    )

            metric_logger.update(loss=output["total_loss"].item())

        evaluation_results = {}
        evaluation_results["wer"] = 200

        for hyp_name in results[name].keys():
            if "gls_hyp" not in hyp_name:
                continue
            k = hyp_name.replace("gls_hyp", "")
            gls_ref = [results[n]["gls_ref"] for n in results]
            gls_hyp = [results[n][hyp_name] for n in results]
            wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
            # evaluation_results[k + "wer_list"] = wer_results
            evaluation_results[k + "wer"] = wer_results["wer"]
            metric_logger.update(**{k + "wer": wer_results["wer"]})
            evaluation_results["wer"] = min(
                wer_results["wer"], evaluation_results["wer"]
            )
        metric_logger.update(wer=evaluation_results["wer"])
        if results_path is not None:
            with open(results_path, "w") as f:
                json.dump(results, f)
    for k, v in evaluation_results.items():
        print(f"{k}: {v:.3f}")
    print("* Averaged results:", metric_logger)
    print("* DEV loss {losses.global_avg:.3f}".format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
