"""
A generic training script that works with any model and dataset.

Author: Paul-Edouard Sarlin (skydes)
"""

import argparse
import copy
import re
import shutil
import signal
from collections import defaultdict
from pathlib import Path
from pydoc import locate

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import __module_name__, logger, settings
from .datasets import get_dataset
from .eval import run_benchmark
from .models import get_model
from .utils.experiments import get_best_checkpoint, get_last_checkpoint, save_experiment
from .utils.stdout_capturing import capture_outputs
from .utils.tensor import batch_to_device
from .utils.tools import (
    AverageMetric,
    MedianMetric,
    PRMetric,
    RecallMetric,
    fork_rng,
    set_seed,
)

# @TODO: Fix pbar pollution in logs
# @TODO: add plotting during evaluation

default_train_conf = {
    "seed": "???",  # training seed
    "epochs": 1,  # number of epochs
    "optimizer": "adam",  # name of optimizer in [adam, sgd, rmsprop]
    "opt_regexp": None,  # regular expression to filter parameters to optimize
    "optimizer_options": {},  # optional arguments passed to the optimizer
    "lr": 0.001,  # learning rate
    "lr_schedule": {
        "type": None,  # string in {factor, exp, member of torch.optim.lr_scheduler}
        "start": 0,
        "exp_div_10": 0,
        "on_epoch": False,
        "factor": 1.0,
        "options": {},  # add lr_scheduler arguments here
    },
    "lr_scaling": [(100, ["dampingnet.const"])],
    "eval_every_iter": 1000,  # interval for evaluation on the validation set
    "save_every_iter": 5000,  # interval for saving the current checkpoint
    "log_every_iter": 200,  # interval for logging the loss to the console
    "log_grad_every_iter": None,  # interval for logging gradient hists
    "test_every_epoch": 1,  # interval for evaluation on the test benchmarks
    "keep_last_checkpoints": 10,  # keep only the last X checkpoints
    "load_experiment": None,  # initialize the model from a previous experiment
    "median_metrics": [],  # add the median of some metrics
    "recall_metrics": {},  # add the recall of some metrics
    "pr_metrics": {},  # add pr curves, set labels/predictions/mask keys
    "best_key": "loss/total",  # key to use to select the best checkpoint
    "dataset_callback_fn": None,  # data func called at the start of each epoch
    "dataset_callback_on_val": False,  # call data func on val data?
    "clip_grad": None,
    "pr_curves": {},
    "plot": None,
    "submodules": [],
    "freeze_2d": False, # whether to freeze the 2D weights when training LightGlu3D
    "initial_3d_with_2d": False, # whether to initialize the 3D weights with the 2D pretrained weight when training LightGlu3D
}
default_train_conf = OmegaConf.create(default_train_conf)


@torch.no_grad()
def do_evaluation(model, loader, device, loss_fn, conf, rank, pbar=True):
    model.eval()
    results = {}
    pr_metrics = defaultdict(PRMetric)
    figures = []
    if conf.plot is not None:
        n, plot_fn = conf.plot
        plot_ids = np.random.choice(len(loader), min(len(loader), n), replace=False)
    for i, data in enumerate(
        tqdm(loader, desc="Evaluation", ascii=True, disable=not pbar)
    ):
        data = batch_to_device(data, device, non_blocking=True)
        with torch.no_grad():
            pred = model(data)
            losses, metrics = loss_fn(pred, data)
            if conf.plot is not None and i in plot_ids:
                figures.append(locate(plot_fn)(pred, data))
            # add PR curves
            for k, v in conf.pr_curves.items():
                pr_metrics[k].update(
                    pred[v["labels"]],
                    pred[v["predictions"]],
                    mask=pred[v["mask"]] if "mask" in v.keys() else None,
                )
            del pred, data
        numbers = {**metrics, **{"loss/" + k: v for k, v in losses.items()}}
        for k, v in numbers.items():
            if k not in results:
                results[k] = AverageMetric()
                if k in conf.median_metrics:
                    results[k + "_median"] = MedianMetric()
                if k in conf.recall_metrics.keys():
                    q = conf.recall_metrics[k]
                    results[k + f"_recall{int(q)}"] = RecallMetric(q)
            results[k].update(v)
            if k in conf.median_metrics:
                results[k + "_median"].update(v)
            if k in conf.recall_metrics.keys():
                q = conf.recall_metrics[k]
                results[k + f"_recall{int(q)}"].update(v)
        del numbers
    results = {k: results[k].compute() for k in results}
    pr_metrics = {k: v.compute() for k, v in pr_metrics.items()}
    return results, pr_metrics, figures


def filter_parameters(params, regexp):
    """Filter trainable parameters based on regular expressions."""

    # Examples of regexp:
    #     '.*(weight|bias)$'
    #     'cnn\.(enc0|enc1).*bias'
    def filter_fn(x):
        n, p = x
        match = re.search(regexp, n)
        if not match:
            p.requires_grad = False
        return match

    params = list(filter(filter_fn, params))
    assert len(params) > 0, regexp
    logger.info("Selected parameters:\n" + "\n".join(n for n, p in params))
    return params


def get_lr_scheduler(optimizer, conf):
    """Get lr scheduler specified by conf.train.lr_schedule."""
    if conf.type not in ["factor", "exp", None]:
        if hasattr(conf.options, "schedulers"):
            # Add option to chain multiple schedulers together
            # This is useful for e.g. warmup, then cosine decay
            schedulers = []
            for scheduler_conf in conf.options.schedulers:
                scheduler = get_lr_scheduler(optimizer, scheduler_conf)
                schedulers.append(scheduler)

            options = {k: v for k, v in conf.options.items() if k != "schedulers"}
            return getattr(torch.optim.lr_scheduler, conf.type)(
                optimizer, schedulers, **options
            )

        return getattr(torch.optim.lr_scheduler, conf.type)(optimizer, **conf.options)

    # backward compatibility
    def lr_fn(it):  # noqa: E306
        if conf.type is None:
            return 1
        if conf.type == "factor":
            return 1.0 if it < conf.start else conf.factor
        if conf.type == "exp":
            gam = 10 ** (-1 / conf.exp_div_10)
            return 1.0 if it < conf.start else gam
        else:
            raise ValueError(conf.type)

    return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)


def pack_lr_parameters(params, base_lr, lr_scaling):
    """Pack each group of parameters with the respective scaled learning rate."""
    filters, scales = tuple(zip(*[(n, s) for s, names in lr_scaling for n in names]))
    scale2params = defaultdict(list)
    for n, p in params:
        scale = 1
        # TODO: use proper regexp rather than just this inclusion check
        is_match = [f in n for f in filters]
        if any(is_match):
            scale = scales[is_match.index(True)]
        scale2params[scale].append((n, p))
    logger.info(
        "Parameters with scaled learning rate:\n%s",
        {s: [n for n, _ in ps] for s, ps in scale2params.items() if s != 1},
    )
    lr_params = [
        {"lr": scale * base_lr, "params": [p for _, p in ps]}
        for scale, ps in scale2params.items()
    ]
    return lr_params


def write_dict_summaries(writer, name, items, step):
    for k, v in items.items():
        key = f"{name}/{k}"
        if isinstance(v, dict):
            writer.add_scalars(key, v, step)
        elif isinstance(v, tuple):
            writer.add_pr_curve(key, *v, step)
        else:
            writer.add_scalar(key, v, step)


def write_image_summaries(writer, name, figures, step):
    if isinstance(figures, list):
        for i, figs in enumerate(figures):
            for k, fig in figs.items():
                writer.add_figure(f"{name}/{i}_{k}", fig, step)
    else:
        for k, fig in figures.items():
            writer.add_figure(f"{name}/{k}", fig, step)

def initialize_3d_and_freeze_2d(model, init_cp=None, freeze_2d=False, initial_3d_with_2d=True, is_restoring=False):
    m = model.module if hasattr(model, "module") else model
    if not is_restoring:
        if init_cp is not None:
            sd = init_cp["model"]
            for i, layer in enumerate(m.transformers):
                qk_w_key = f"cross_attn.{i}.to_qk.weight"
                qk_b_key = f"cross_attn.{i}.to_qk.bias"
                v_w_key = f"cross_attn.{i}.to_v.weight"
                v_b_key = f"cross_attn.{i}.to_v.bias"
                
                if qk_w_key in sd:
                    with torch.no_grad():
                        try:
                            layer.cross_attn.to_qk2d.weight.copy_(sd[qk_w_key])
                            layer.cross_attn.to_qk2d.bias.copy_(sd[qk_b_key])
                            layer.cross_attn.to_v2d.weight.copy_(sd[v_w_key])
                            layer.cross_attn.to_v2d.bias.copy_(sd[v_b_key])
                            if i == 0:
                                print(f"Initialized 2D-source cross-attn weights.")
                        except Exception as e:
                            print(f"Failed to initialize 2D-source weights for layer {i}: {e}")
                    
                        if initial_3d_with_2d:
                            try:
                                layer.self_attn3d.load_state_dict(layer.self_attn.state_dict())
                                layer.cross_attn.to_qk3d.weight.copy_(sd[qk_w_key])
                                layer.cross_attn.to_qk3d.bias.copy_(sd[qk_b_key])
                                layer.cross_attn.to_v3d.weight.copy_(sd[v_w_key])
                                layer.cross_attn.to_v3d.bias.copy_(sd[v_b_key])
                                if i==0:
                                    print(f"Initialized 3D-source cross-attn weights with 2D weights.")
                            except Exception as e:
                                print(f"Failed to initialize 3D-source weights with 2D weights for layer {i}: {e}")

                else:
                    print(f"WARNING: qk_w_key {qk_w_key} not found in layer{i}.")
        else:
            print("WARNING: Failed to initialize weights.")
        if initial_3d_with_2d:
            dict_3d = m.posenc3d.state_dict()
            for name, param in m.posenc.state_dict().items():
                if name in dict_3d:
                    if param.shape == dict_3d[name].shape:
                        dict_3d[name].copy_(param)
                    elif "weight" in name and param.dim() == 2:
                        in_dim_src = param.shape[1]
                        in_dim_dst = dict_3d[name].shape[1]
                        # copy the 2d posenc weights to the first two dimensions of posenc3d
                        if in_dim_src < in_dim_dst:
                            dict_3d[name][:, :in_dim_src].copy_(param)
                            print(f"Copy for {name}: {in_dim_src} dims copied to {in_dim_dst} dims.")
                        else:
                            print(f"Skipping {name} due to incompatible shapes.")
            m.posenc3d.load_state_dict(dict_3d)
        else: 
             print("Skipping 3D posenc weights initialization (not enabled in config).")
            
    else:
        print("Restore mode: Using weights from checkpoint.")
    
    if freeze_2d:
        for name, param in m.named_parameters():
            if any(x in name for x in ["self_attn.", "posenc.", "cross_attn.to_qk2d", "cross_attn.to_v2d"]) and "3d" not in name:
                param.requires_grad = False
        print("Frozen 2D-source projection layers, allowing 3D-source layers to adapt.")
    else:
        for name, param in m.named_parameters():
            param.requires_grad = True
        print("2D weights are UNFROZEN.")
def training(rank, conf, output_dir, args):
    if args.restore:
        with open_dict(conf):
            conf.model.is_restoring = True
        logger.info(f"Restoring from previous training of {args.experiment}")
        try:
            init_cp = get_last_checkpoint(args.experiment, allow_interrupted=False)
        except AssertionError:
            init_cp = get_best_checkpoint(args.experiment)
        logger.info(f"Restoring from checkpoint {init_cp.name}")
        # init_cp = torch.load(
        #     str(init_cp), map_location="cpu", weights_only=not settings.ALLOW_PICKLE
        # )
        init_cp = torch.load(
            str(init_cp), map_location="cpu", weights_only=False
        )
        conf = OmegaConf.merge(OmegaConf.create(init_cp["conf"]), conf)
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        epoch = init_cp["epoch"] + 1

        # get the best loss or eval metric from the previous best checkpoint
        best_cp = get_best_checkpoint(args.experiment)
        # best_cp = torch.load(
        #     str(best_cp), map_location="cpu", weights_only=not settings.ALLOW_PICKLE
        # )
        best_cp = torch.load(
            str(best_cp), map_location="cpu", weights_only=False
        )
        best_eval = best_cp["eval"][conf.train.best_key]
        del best_cp
    else:
        # we start a new, fresh training
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        epoch = 0
        best_eval = float("inf")
        if conf.train.load_experiment:
            logger.info(f"Will fine-tune from weights of {conf.train.load_experiment}")
            # the user has to make sure that the weights are compatible
            try:
                init_cp = get_last_checkpoint(conf.train.load_experiment)
            except AssertionError:
                init_cp = get_best_checkpoint(conf.train.load_experiment)
            # init_cp = get_last_checkpoint(conf.train.load_experiment)
            init_cp = torch.load(
                str(init_cp), map_location="cpu", weights_only=not settings.ALLOW_PICKLE
            )
            # load the model config of the old setup, and overwrite with current config
            conf.model = OmegaConf.merge(
                OmegaConf.create(init_cp["conf"]).model, conf.model
            )
            print(conf.model)
        else:
            init_cp = None

    OmegaConf.set_struct(conf, True)  # prevent access to unknown entries
    set_seed(conf.train.seed)
    if rank == 0:
        writer = SummaryWriter(log_dir=str(output_dir))

    data_conf = copy.deepcopy(conf.data)
    if args.distributed:
        logger.info(f"Training in distributed mode with {args.n_gpus} GPUs")
        assert torch.cuda.is_available()
        device = rank
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=args.n_gpus,
            rank=device,
            init_method="file://" + str(args.lock_file),
        )
        torch.cuda.set_device(device)

        # adjust batch size and num of workers since these are per GPU
        if "batch_size" in data_conf:
            data_conf.batch_size = int(data_conf.batch_size / args.n_gpus)
        if "train_batch_size" in data_conf:
            data_conf.train_batch_size = int(data_conf.train_batch_size / args.n_gpus)
        if "num_workers" in data_conf:
            data_conf.num_workers = int(
                (data_conf.num_workers + args.n_gpus - 1) / args.n_gpus
            )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device {device}")

    dataset = get_dataset(data_conf.name)(data_conf)

    # Optionally load a different validation dataset than the training one
    val_data_conf = conf.get("data_val", None)
    if val_data_conf is None:
        val_dataset = dataset
    else:
        val_dataset = get_dataset(val_data_conf.name)(val_data_conf)

    # @TODO: add test data loader

    if args.overfit:
        # we train and eval with the same single training batch
        logger.info("Data in overfitting mode")
        assert not args.distributed
        train_loader = dataset.get_overfit_loader("train")
        val_loader = val_dataset.get_overfit_loader("val")
    else:
        train_loader = dataset.get_data_loader("train", distributed=args.distributed, pinned=True)
        val_loader = val_dataset.get_data_loader("val", pinned=True)
    if rank == 0:
        logger.info(f"Training loader has {len(train_loader)} batches")
        logger.info(f"Validation loader has {len(val_loader)} batches")

    # interrupts are caught and delayed for graceful termination
    def sigint_handler(signal, frame):
        logger.info("Caught keyboard interrupt signal, will terminate")
        nonlocal stop
        if stop:
            raise KeyboardInterrupt
        stop = True

    stop = False
    signal.signal(signal.SIGINT, sigint_handler)
    model = get_model(conf.model.name)(conf.model).to(device)   
    if init_cp is not None:
        model.load_state_dict(init_cp["model"], strict=False)

    raw_model = model.module if hasattr(model, "module") else model
    if "lightglu3d" in raw_model.__class__.__name__.lower():
        if init_cp == None:
            weights_path = conf.model.get("weights") 
            if weights_path:
                logger.info(f"Loading weights from {weights_path} for manual stitching...")
                init_cp = torch.load(weights_path, map_location="cpu")
                if "model" not in init_cp:
                    init_cp = {"model": init_cp}
                
        initialize_3d_and_freeze_2d(
            model,
            init_cp=init_cp, 
            freeze_2d=conf.train.freeze_2d, 
            initial_3d_with_2d=conf.train.initial_3d_with_2d,
            is_restoring=args.restore
        )

    if args.compile:
        model = torch.compile(model, mode=args.compile)
    loss_fn = model.loss
    # logger for debugging
    if rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        logger.info(f"Model initialized: Trainable={trainable_params/1e6:.2f}M, Frozen={frozen_params/1e6:.2f}M")
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device],find_unused_parameters=conf.train.get("freeze_2d", False)
            )
    if rank == 0 and args.print_arch:
        logger.info(f"Model: \n{model}")

    torch.backends.cudnn.benchmark = True
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    optimizer_fn = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }[conf.train.optimizer]
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if conf.train.opt_regexp:
        params = filter_parameters(params, conf.train.opt_regexp)
    all_params = [p for n, p in params]

    lr_params = pack_lr_parameters(params, conf.train.lr, conf.train.lr_scaling)
    optimizer = optimizer_fn(
        lr_params, lr=conf.train.lr, **conf.train.optimizer_options
    )

    for i, g in enumerate(optimizer.param_groups):
        if g['lr'] != conf.train.lr:
            print(f"Group {i} with lr at {g['lr']}")
            
    use_mp = args.mixed_precision is not None
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_mp)
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=use_mp)
    )
    logger.info(f"Training with mixed_precision={args.mixed_precision}")

    mp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        None: torch.float32,  # we disable it anyway
    }[args.mixed_precision]

    results = None  # fix bug with it saving

    lr_scheduler = get_lr_scheduler(optimizer=optimizer, conf=conf.train.lr_schedule)
    if args.restore:
        try:
            optimizer.load_state_dict(init_cp["optimizer"])
        except ValueError as e:
            print(f"Warning: Optimizer shape mismatch ({e}). Initializing fresh optimizer state.")
        if "lr_scheduler" in init_cp:
            lr_scheduler.load_state_dict(init_cp["lr_scheduler"])

    if rank == 0:
        logger.info(
            "Starting training with configuration:\n%s", OmegaConf.to_yaml(conf)
        )

    def trace_handler(p):
        # torch.profiler.tensorboard_trace_handler(str(output_dir))
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(output)
        p.export_chrome_trace("trace_" + str(p.step_num) + ".json")
        p.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")

    if args.profile:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.__enter__()
    while epoch < conf.train.epochs and not stop:
        if rank == 0:
            logger.info(f"Starting epoch {epoch}")

        # we first run the eval
        if (
            rank == 0
            and epoch % conf.train.test_every_epoch == 0
            and args.run_benchmarks
        ):
            for bname, eval_conf in conf.get("benchmarks", {}).items():
                logger.info(f"Running eval on {bname}")
                summaries, figures, _ = run_benchmark(
                    bname,
                    eval_conf,
                    settings.EVAL_PATH / bname / args.experiment / str(epoch),
                    model.eval(),
                )
                str_summaries = [
                    f"{k} {v:.3E}" for k, v in summaries.items() if isinstance(v, float)
                ]
                logger.info(f'[{bname}] {{{", ".join(str_summaries)}}}')
                write_dict_summaries(writer, f"test/{bname}", summaries, epoch)
                write_image_summaries(writer, f"figures/{bname}", figures, epoch)
                del summaries, figures

        # set the seed
        set_seed(conf.train.seed + epoch)

        # update learning rate
        if conf.train.lr_schedule.on_epoch and epoch > 0:
            old_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step()
            logger.info(
                f'lr changed from {old_lr} to {optimizer.param_groups[0]["lr"]}'
            )
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        if epoch > 0 and conf.train.dataset_callback_fn and not args.overfit:
            loaders = [train_loader]
            if conf.train.dataset_callback_on_val:
                loaders += [val_loader]
            for loader in loaders:
                if isinstance(loader.dataset, torch.utils.data.Subset):
                    getattr(loader.dataset.dataset, conf.train.dataset_callback_fn)(
                        conf.train.seed + epoch
                    )
                else:
                    getattr(loader.dataset, conf.train.dataset_callback_fn)(
                        conf.train.seed + epoch
                    )
        for it, data in enumerate(train_loader):
            tot_it = (len(train_loader) * epoch + it) * (
                args.n_gpus if args.distributed else 1
            )
            tot_n_samples = tot_it
            if not args.log_it:
                # We normalize the x-axis of tensorflow to num samples!
                tot_n_samples *= train_loader.batch_size

            model.train()
            optimizer.zero_grad()

            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                enabled=args.mixed_precision is not None,
                dtype=mp_dtype,
            ):
                data = batch_to_device(data, device, non_blocking=True)
                pred = model(data)
                losses, _ = loss_fn(pred, data)
                loss = torch.mean(losses["total"])
            if torch.isnan(loss).any():
                print(f"Detected NAN, skipping iteration {it}")
                del pred, data, loss, losses
                continue

            do_backward = loss.requires_grad
            if args.distributed:
                do_backward = torch.tensor(do_backward).float().to(device)
                torch.distributed.all_reduce(
                    do_backward, torch.distributed.ReduceOp.PRODUCT
                )
                do_backward = do_backward > 0
            if do_backward:
                scaler.scale(loss).backward()
                if args.detect_anomaly:
                    # Check for params without any gradient which causes
                    # problems in distributed training with checkpointing
                    detected_anomaly = False
                    for name, param in model.named_parameters():
                        if param.grad is None and param.requires_grad:
                            print(f"param {name} has no gradient.")
                            detected_anomaly = True
                    if detected_anomaly:
                        raise RuntimeError("Detected anomaly in training.")
                if conf.train.get("clip_grad", None):
                    scaler.unscale_(optimizer)
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            all_params,
                            max_norm=conf.train.clip_grad,
                            error_if_nonfinite=True,
                        )
                        scaler.step(optimizer)
                    except RuntimeError:
                        logger.warning("NaN detected in gradients. Skipping iteration.")
                    scaler.update()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                if not conf.train.lr_schedule.on_epoch:
                    lr_scheduler.step()
            else:
                if rank == 0:
                    logger.warning(f"Skip iteration {it} due to detach.")

            if args.profile:
                prof.step()

            if it % conf.train.log_every_iter == 0:
                for k in sorted(losses.keys()):
                    if args.distributed:
                        losses[k] = losses[k].sum(-1)
                        torch.distributed.reduce(losses[k], dst=0)
                        losses[k] /= train_loader.batch_size * args.n_gpus
                    losses[k] = torch.mean(losses[k], -1)
                    losses[k] = losses[k].item()
                if rank == 0:
                    str_losses = [f"{k} {v:.3E}" for k, v in losses.items()]
                    logger.info(
                        "[E {} | it {}] loss {{{}}}".format(
                            epoch, it, ", ".join(str_losses)
                        )
                    )
                    write_dict_summaries(writer, "training/", losses, tot_n_samples)
                    writer.add_scalar(
                        "training/lr", optimizer.param_groups[0]["lr"], tot_n_samples
                    )
                    writer.add_scalar("training/epoch", epoch, tot_n_samples)

            if conf.train.log_grad_every_iter is not None:
                if it % conf.train.log_grad_every_iter == 0:
                    grad_txt = ""
                    for name, param in model.named_parameters():
                        if param.grad is not None and param.requires_grad:
                            if name.endswith("bias"):
                                continue
                            writer.add_histogram(
                                f"grad/{name}", param.grad.detach(), tot_n_samples
                            )
                            norm = torch.norm(param.grad.detach(), 2)
                            grad_txt += f"{name} {norm.item():.3f}  \n"
                    writer.add_text("grad/summary", grad_txt, tot_n_samples)
            del pred, data, loss, losses

            # Run validation
            if (
                (
                    it % conf.train.eval_every_iter == 0
                    and (it > 0 or epoch == -int(args.no_eval_0))
                )
                or stop
                or it == (len(train_loader) - 1)
            ):
                with fork_rng(seed=conf.train.seed):
                    results, pr_metrics, figures = do_evaluation(
                        model,
                        val_loader,
                        device,
                        loss_fn,
                        conf.train,
                        rank,
                        pbar=(rank == 0),
                    )

                if rank == 0:
                    str_results = [
                        f"{k} {v:.3E}"
                        for k, v in results.items()
                        if isinstance(v, float)
                    ]
                    logger.info(f'[Validation] {{{", ".join(str_results)}}}')
                    write_dict_summaries(writer, "val", results, tot_n_samples)
                    write_dict_summaries(writer, "val", pr_metrics, tot_n_samples)
                    write_image_summaries(writer, "figures", figures, tot_n_samples)
                    # @TODO: optional always save checkpoint
                    if results[conf.train.best_key] < best_eval:
                        best_eval = results[conf.train.best_key]
                        save_experiment(
                            model,
                            optimizer,
                            lr_scheduler,
                            conf,
                            results,
                            best_eval,
                            epoch,
                            tot_it,
                            output_dir,
                            stop,
                            args.distributed,
                            cp_name="checkpoint_best.tar",
                        )
                        logger.info(f"New best val: {conf.train.best_key}={best_eval}")
                torch.cuda.empty_cache()  # should be cleared at the first iter

            if (tot_it % conf.train.save_every_iter == 0 and tot_it > 0) and rank == 0:
                if results is None:
                    results, _, _ = do_evaluation(
                        model,
                        val_loader,
                        device,
                        loss_fn,
                        conf.train,
                        rank,
                        pbar=(rank == 0),
                    )
                    best_eval = results[conf.train.best_key]
                best_eval = save_experiment(
                    model,
                    optimizer,
                    lr_scheduler,
                    conf,
                    results,
                    best_eval,
                    epoch,
                    tot_it,
                    output_dir,
                    stop,
                    args.distributed,
                )
            if stop:
                break

        if rank == 0:
            best_eval = save_experiment(
                model,
                optimizer,
                lr_scheduler,
                conf,
                results,
                best_eval,
                epoch,
                tot_it,
                output_dir=output_dir,
                stop=stop,
                distributed=args.distributed,
            )

        results = None  # free memory
        epoch += 1

    logger.info(f"Finished training on process {rank}.")
    if rank == 0:
        writer.close()


def main_worker(rank, conf, output_dir, args):
    if rank == 0:
        with capture_outputs(
            output_dir / "log.txt", cleanup_interval=args.cleanup_interval
        ):
            training(rank, conf, output_dir, args)
    else:
        training(rank, conf, output_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--conf", type=str)
    parser.add_argument(
        "--mixed_precision",
        "--mp",
        default=None,
        type=str,
        choices=["float16", "bfloat16"],
    )
    parser.add_argument(
        "--compile",
        default=None,
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument(
        "--cleanup_interval",
        default=120,  # Cleanup log files every 120 seconds.
        type=int,
    )
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--print_arch", "--pa", action="store_true")
    parser.add_argument("--detect_anomaly", "--da", action="store_true")
    parser.add_argument("--log_it", "--log_it", action="store_true")
    parser.add_argument("--no_eval_0", action="store_true")
    parser.add_argument("--run_benchmarks", action="store_true")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()

    logger.info(f"Starting experiment {args.experiment}")
    output_dir = Path(settings.TRAINING_PATH, args.experiment)
    output_dir.mkdir(exist_ok=True, parents=True)

    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        yaml_conf = OmegaConf.load(args.conf)
        OmegaConf.resolve(yaml_conf)
        conf = OmegaConf.merge(yaml_conf, conf)
    elif args.restore:
        restore_conf = OmegaConf.load(output_dir / "config.yaml")
        conf = OmegaConf.merge(restore_conf, conf)
    if not args.restore:
        if conf.train.seed is None:
            conf.train.seed = torch.initial_seed() & (2**32 - 1)
        OmegaConf.save(conf, str(output_dir / "config.yaml"))

    # copy gluefactory and submodule into output dir
    for module in conf.train.get("submodules", []) + [__module_name__]:
        mod_dir = Path(__import__(str(module)).__file__).parent
        shutil.copytree(mod_dir, output_dir / module, dirs_exist_ok=True)
    if args.distributed:
        args.n_gpus = torch.cuda.device_count()
        args.lock_file = output_dir / "distributed_lock"
        if args.lock_file.exists():
            args.lock_file.unlink()
        torch.multiprocessing.spawn(
            main_worker, nprocs=args.n_gpus, args=(conf, output_dir, args)
        )
    else:
        main_worker(0, conf, output_dir, args)
