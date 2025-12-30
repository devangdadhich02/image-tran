import collections
import json
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from domainbed.datasets import get_dataset, split_dataset
from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import swad as swad_module


def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def train(test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None):
    logger.info("")

    #######################################################
    # setup dataset & loader
    #######################################################
    args.real_test_envs = test_envs  # for log
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)
    test_splits = []

    if target_env is not None:
        testenv_name = f"te_{dataset.environments[target_env]}"
        logger.info(f"Target env = {target_env}")
    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = "te_" + "_".join(testenv_properties)

    logger.info(
        "Testenv name escaping {} -> {}".format(testenv_name, testenv_name.replace(".", ""))
    )
    testenv_name = testenv_name.replace(".", "")
    logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

    n_envs = len(dataset)
    train_envs = sorted(set(range(n_envs)) - set(test_envs))
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=np.int)

    batch_sizes[test_envs] = 0
    batch_sizes = batch_sizes.tolist()

    logger.info(f"Batch sizes for each domain: {batch_sizes} (total={sum(batch_sizes)})")

    # calculate steps per epoch
    steps_per_epochs = [
        len(env) / batch_size
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    steps_per_epoch = min(steps_per_epochs)
    # epoch is computed by steps_per_epoch
    # cutting these out to simplify datasets so that we can train quicker.
    prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
    logger.info(f"steps-per-epoch for each domain: {prt_steps} -> min = {steps_per_epoch:.2f}")
    
    #n_steps = int(steps_per_epoch / 4)
    #logger.info(f"!!! DEBUG MODE: Reducing n_steps to {n_steps} (1/4 of epoch) for quick reconstruction check !!!")
    # new code: overwrite n_steps to be exactly 1/3 of an epoch.
    


    # setup loaders
    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=batch_size,
            num_workers=dataset.N_WORKERS,
        )
        for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]

    # setup eval loaders
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(in_splits + out_splits + test_splits):
        batchsize = hparams["test_batchsize"]
        loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
        if args.prebuild_loader:
            loader_kwargs = FastDataLoader(**loader_kwargs)
        eval_loaders_kwargs.append(loader_kwargs)

    eval_weights = [None for _, weights in (in_splits + out_splits + test_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_loader_names += ["env{}_inTE".format(i) for i in range(len(test_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))

    #######################################################
    # setup algorithm (model)
    #######################################################
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )

    algorithm.cuda()

    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    #######################################################
    # start training loop
    #######################################################
    evaluator = Evaluator(
        test_envs,
        eval_meta,
        n_envs,
        logger,
        evalmode=args.evalmode,
        debug=args.debug,
        target_env=target_env,
    )

    swad = None
    if hparams["swad"]:
        swad_algorithm = swa_utils.AveragedModel(algorithm)
        swad_cls = getattr(swad_module, "LossValley")
        swad = swad_cls(evaluator, **hparams.swad_kwargs)

    # ================= NEW: checkpoint + recon saving =================
    # --- checkpoint saving config ---
    SAVE_INTERVAL = 200

    # Save first K checkpoints
    FIRST_K = 3
    first_ckpt_count = 0

    # Save last K checkpoints (rolling)
    LAST_K = 3
    last_ckpts = collections.deque()

    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    last_results_keys = None
    records = []
    epochs_path = args.out_dir / "results.jsonl"

    for step in range(n_steps):
        step_start_time = time.time()
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
        batches_dictlist = next(train_minibatches_iterator)
        # batches: {data_key: [env0_tensor, ...], ...}
        batches = misc.merge_dictlist(batches_dictlist)
        # to device
        batches = {
            key: [tensor.cuda() for tensor in tensorlist] for key, tensorlist in batches.items()
        }

        inputs = {**batches, "step": step}
        step_vals = algorithm.update(**inputs)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        if swad:
            # swad_algorithm is segment_swa for swad
            swad_algorithm.update_parameters(algorithm, step=step)

        if step % checkpoint_freq == 0:
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            eval_start_time = time.time()
            accuracies, summaries = evaluator.evaluate(algorithm)

            results["eval_time"] = time.time() - eval_start_time

            # results = (epochs, loss, step, step_time)
            results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
            # merge results
            results.update(summaries)
            results.update(accuracies)

            # print
            if results_keys != last_results_keys:
                logger.info(misc.to_row(results_keys))
                last_results_keys = results_keys
            logger.info(misc.to_row([results[key] for key in results_keys]))

            if step != 0:
                max_train_out = max(records, key=lambda x: x['train_out'])['train_out'] #added this line
            records.append(copy.deepcopy(results))

            # update results to record
            results.update({"hparams": dict(hparams), "args": vars(args)})

            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

            checkpoint_vals = collections.defaultdict(lambda: [])

            writer.add_scalars_with_prefix(summaries, step, f"{testenv_name}/summary/")
            writer.add_scalars_with_prefix(accuracies, step, f"{testenv_name}/all/")

            if args.model_save and step >= args.model_save:
                ckpt_dir = args.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                if results["train_out"] > max_train_out or step == n_steps-1:    
                    test_env_str = ",".join(map(str, test_envs))
                    if step == n_steps-1:
                        filename = "TE{}_{}.pth".format(test_env_str, step)
                    else:
                        filename = "TE{}.pth".format(test_env_str)

                    if len(test_envs) > 1 and target_env is not None:
                        train_env_str = ",".join(map(str, train_envs))
                        filename = f"TE{target_env}_TR{train_env_str}_{step}.pth"
                    path = ckpt_dir / filename

                    save_dict = {
                        "args": vars(args),
                        "model_hparams": dict(hparams),
                        "test_envs": test_envs,
                        "model_dict": algorithm.cpu().state_dict(),
                    }
                    algorithm.cuda()
                    if not args.debug:
                        torch.save(save_dict, path)
                    else:
                        logger.debug("DEBUG Mode -> no save (org path: %s)" % path)

            # swad
            if swad:
                def prt_results_fn(results, avgmodel):
                    step_str = f" [{avgmodel.start_step}-{avgmodel.end_step}]"
                    row = misc.to_row([results[key] for key in results_keys if key in results])
                    logger.info(row + step_str)

                swad.update_and_evaluate(
                    swad_algorithm, results["train_out"], results["tr_outloss"], prt_results_fn
                )

                if hasattr(swad, "dead_valley") and swad.dead_valley:
                    logger.info("SWAD valley is dead -> early stop !")
                    break

                swad_algorithm = swa_utils.AveragedModel(algorithm)  # reset

        if step % args.tb_freq == 0:
            # add step values only for tb log
            writer.add_scalars_with_prefix(step_vals, step, f"{testenv_name}/summary/")


    # ---------------------------------------------------------

        if step > 0 and (step % SAVE_INTERVAL) == 0:
            test_env_str = ",".join(map(str, test_envs))
            ckpt_name = f"TE{test_env_str}_step{step}.pth"
            ckpt_path = ckpt_dir / ckpt_name

            save_dict = {
                "args": vars(args),
                "model_hparams": dict(hparams),
                "test_envs": test_envs,
                "model_dict": algorithm.cpu().state_dict(),
                "step": step,
            }

            # ---- EARLY PHASE: save first 3 checkpoints ----
            if first_ckpt_count < FIRST_K:
                try:
                    if not args.debug:
                        torch.save(save_dict, ckpt_path)
                    else:
                        ckpt_path.with_suffix(".debug").write_text("debug checkpoint\n")
                finally:
                    try:
                        algorithm.cuda()
                    except Exception:
                        pass

                first_ckpt_count += 1
                logger.nofmt(
                    f"[SAVE] early checkpoint {first_ckpt_count}/{FIRST_K} at step {step}"
                )

            # ---- LATE PHASE: rolling save last 3 checkpoints ----
            else:
                try:
                    if not args.debug:
                        torch.save(save_dict, ckpt_path)
                    else:
                        ckpt_path.with_suffix(".debug").write_text("debug checkpoint\n")
                finally:
                    try:
                        algorithm.cuda()
                    except Exception:
                        pass

                # manage rolling buffer
                if len(last_ckpts) >= LAST_K:
                    old = last_ckpts.popleft()
                    try:
                        if old.exists():
                            old.unlink()
                    except Exception as e:
                        logger.nofmt(
                            f"Warning deleting old late checkpoint {old}: {e}"
                        )

                last_ckpts.append(ckpt_path)
                logger.nofmt(
                    f"[SAVE] late checkpoint (rolling) at step {step}"
                )

            # [!!! CHANGED SECTION STARTS HERE !!!]
            # ---- Save reconstruction images ----
            recon_save_dir = args.out_dir / f"recon_step_{step}"
            try:
                # 1. Switch to eval mode
                # This freezes Batch Normalization statistics so the reconstruction pass
                # does not corrupt the training model's running mean/variance.
                algorithm.eval() 
                
                # 2. Disable gradients
                # This ensures PyTorch doesn't track operations for backprop, 
                # saving memory and guaranteeing weights cannot be updated.
                with torch.no_grad():
                    algorithm.save_final_reconstruction(
                        batches, save_dir=recon_save_dir
                    )
            except Exception as e:
                logger.nofmt(
                    f"Warning: failed to save reconstruction at step {step}: {e}"
                )
            finally:
                # 3. Switch back to train mode
                # This is inside 'finally' to ensure it runs even if the save fails.
                # If we forget this, the model stays frozen for the rest of training!
                algorithm.train()
            # [!!! CHANGED SECTION ENDS HERE !!!]
        
        # ==============================================================


        if step == n_steps - 1:
            # Also applied the protection to the final save
            try:
                algorithm.eval()
                with torch.no_grad():
                    algorithm.save_final_reconstruction(
                        batches, 
                        save_dir=args.out_dir / "recon_final"
                    )
            finally:
                algorithm.train()



    # find best
    logger.info("---")
    records = Q(records)
    te_val_best = records.argmax("test_out")["test_in"]
    tr_val_best = records.argmax("train_out")["test_in"]
    last = records[-1]["test_in"]

    in_key = "train_out"
    tr_val_best_indomain = records.argmax("train_out")[in_key]
    last_indomain = records[-1][in_key]

    # NOTE We report only training-domain validation results, oracle results are used just as an upperbound not for direct comparison.
    ret = {
        "test-domain validation": te_val_best,
        "training-domain validation": tr_val_best,
        #  "last": last,
        #  "last (inD)": last_indomain,
        #  "training-domain validation (inD)": tr_val_best_indomain,
    }

    # Evaluate SWAD
    if swad:
        swad_algorithm = swad.get_final_model()
        if hparams["freeze_bn"] is False:
            n_steps = 500 if not args.debug else 10
            logger.warning(f"Update SWAD BN statistics for {n_steps} steps ...")
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps)

        logger.warning("Evaluate SWAD ...")
        accuracies, summaries = evaluator.evaluate(swad_algorithm)
        results = {**summaries, **accuracies}
        start = swad_algorithm.start_step
        end = swad_algorithm.end_step
        step_str = f" [{start}-{end}]  (N={swad_algorithm.n_averaged})"
        row = misc.to_row([results[key] for key in results_keys if key in results]) + step_str
        logger.info(row)

        ret["SWAD"] = results["test_in"]
        ret["SWAD (inD)"] = results[in_key]

    for k, acc in ret.items():
        logger.info(f"{k} = {acc:.3%}")

    return ret, records
