
import torch
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path

def get_logger(args, gpu_id=0):
    logging.basicConfig(level=logging.INFO)
    name_logfile = "logFile_fineTune" if args.prune_ratio is None else "logFile_prune"
    file_logging = os.path.join(args.output_dir, name_logfile + ".log")
    logger = logging.getLogger("logger")
    if gpu_id > 0:
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.INFO)
    # self.logger.setLevel(self.cfg.logging.log_level)

    file_mode = 'a' if len(args.resume) > 0 else 'w'
    file_handler = logging.FileHandler(file_logging, mode=file_mode)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(args)
    logger.info("output Dir: %s" % args.output_dir)

    return logger

df_imp_criteria=None

def log_model_imp_criteria(model, epoch, save_results, args):
    global df_imp_criteria
    if df_imp_criteria is None:
        df_imp_criteria=pd.DataFrame({"mag":np.zeros(24), "grad":np.zeros(24), "taylorFo":np.zeros(24), "cnt":np.zeros(24)})
    for name, p in model.named_parameters():
        if "blocks" in name and "mask" not in name:
            block_no = int(name.split(".")[1])
            if "norm1" in name or "attn" in name:
                block_sub_no = 0
            elif "norm2" in name or "mlp" in name:
                block_sub_no = 1
            else:
                assert False
            imp_idx = 2*block_no + block_sub_no

            grad_inf_mask = torch.logical_not(torch.isinf(p.grad.data))
            mag = p.data.abs().sum().item()
            grad = p.grad.data[grad_inf_mask].abs().sum().item()
            taylorFo = (p.data * p.grad.data)[grad_inf_mask].pow(2).sum().item() * 1e6
            cnt = p.data.numel()

            df_imp_criteria.loc[imp_idx, "mag"] = df_imp_criteria.loc[imp_idx, "mag"] + mag
            df_imp_criteria.loc[imp_idx, "grad"] = df_imp_criteria.loc[imp_idx, "grad"] + grad
            df_imp_criteria.loc[imp_idx, "taylorFo"] = df_imp_criteria.loc[imp_idx, "taylorFo"] + taylorFo
            df_imp_criteria.loc[imp_idx, "cnt"] = df_imp_criteria.loc[imp_idx, "cnt"] + cnt



            y=1

    #imp_criteria["mag"] = imp_criteria["mag"]/imp_criteria["cnt"]
    #imp_criteria["grad"] = imp_criteria["grad"]/imp_criteria["cnt"]
    #imp_criteria["taylorFo"] = 1e10* imp_criteria["taylorFo"]/imp_criteria["cnt"]


    #imp_criteria["mag"] /= imp_criteria["mag"].max()
    #imp_criteria["grad"] /= imp_criteria["grad"].max()
    #imp_criteria["taylorFo"] /= imp_criteria["taylorFo"].max()
    plot_epochs = [0, 5, 10, 30, 50]
    if save_results and epoch in plot_epochs:
        df_imp_criteria[["mag", "grad", "taylorFo"]] = df_imp_criteria[["mag", "grad", "taylorFo"]].div(
            df_imp_criteria["cnt"], axis=0)
        df_imp_criteria = df_imp_criteria.drop(["cnt"], axis=1)

        #add norm
        df_imp_criteria["mag_norm"] = df_imp_criteria["mag"] / df_imp_criteria["mag"].max()
        df_imp_criteria["grad_norm"] = df_imp_criteria["grad"] / df_imp_criteria["grad"].max()
        df_imp_criteria["taylorFo_norm"] = df_imp_criteria["taylorFo"] / df_imp_criteria["taylorFo"].max()

        path_out_file = os.path.join(args.output_dir, "imp_criteria")
        Path(path_out_file).mkdir(parents=True, exist_ok=True)
        path_out_file = os.path.join(path_out_file, f"avg_imp_{epoch}.csv")
        df_imp_criteria.to_csv(path_out_file, sep=",", header=True)
        y=1

        df_imp_criteria = pd.DataFrame(
            {"mag": np.zeros(24), "grad": np.zeros(24), "taylorFo": np.zeros(24), "cnt": np.zeros(24)})

    y = 1

