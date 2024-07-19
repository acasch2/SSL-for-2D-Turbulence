from utils.trainer import Trainer
from utils.YParams import YParams
import torch
import logging
import os
import pickle
import argparse


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", type=str)
    parser.add_argument("--config", type=str)

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)


    if params["log_to_screen"]:
        logging.basicConfig(level=logging.INFO,
                format="%(levelname)s | %(asctime)s | %(message)s",
                filename=params["logfile_path"])
        
    trainer = Trainer(params)
    trainer.train()

    if params["save_model"]:
        model_fp = os.path.normpath(params["model_save_path"])
        model_fp_split = model_fp.split(os.sep)
        model_fp = os.path.join("/",*model_fp_split[:-1], "last_epoch_"+model_fp_split[-1])
        torch.save(trainer.model.state_dict(), model_fp)
        logging.info("Successfully saved model state_dict.")
        with open(params["params_save_path"], "wb") as fp:
            pickle.dump(params, fp)
        logging.info("Successfully saved params file.")
