from utils.trainer import Trainer
import torch
import torch.nn as nn
import logging
import os
import pickle

# Define all parameters
params = {
    "gpu": True,
    "lr" : 1e-4,
    "weight_decay" : 1e-2,
    "max_epochs" : 5,
    "data_dir" : '/ocean/projects/phy220045p/jakhar/py2d_dealias/results/Re5000_fkx4fky4_r0.1_b20.0/NoSGS/NX256/dt0.0002_IC1/data/',
    "train_file_range" : (200000, 220000),
    "valid_file_range" : (310000, 312000),
    "batch_size" : 4,
    "target_step": 3,
    "num_workers" : 1,
    "pin_memory" : True,
    "img_size" : 256,
    "patch_size" : 4,
    "num_frames" : 1,
    "tubelet_size" : 1,
    "in_chans" : 2,
    "encoder_embed_dim" : 192,
    "encoder_depth" : 6,
    "encoder_num_heads" : 6,
    "decoder_embed_dim" : 192,
    "decoder_depth" : 6,
    "decoder_num_heads" : 6,
    "mlp_ratio" : 4.,
    "norm_layer" : nn.LayerNorm,
    "num_out_frames" : 1,
    "log_to_screen" : True,
    "logfile_path": '/ocean/projects/atm170004p/dpatel9/Logfiles/test10.log',
    "save_model": True,
    "model_save_dir": '/ocean/projects/atm170004p/dpatel9/ML_Weights/Base_Emulator/',
    "model_save_name": 'test10.pt',
    "params_save_name": 'test10_parameters.pkl'
}


if params["log_to_screen"]:
    logging.basicConfig(level=logging.INFO,
            format="%(levelname)s | %(asctime)s | %(message)s",
            filename=params["logfile_path"])
    
trainer = Trainer(params)
trainer.train()

if params["save_model"]:
    model_fp = os.path.join(params["model_save_dir"], params["model_save_name"])
    torch.save(trainer.model.state_dict(), model_fp)
    params_fp = os.path.join(params["model_save_dir"], params["params_save_name"])
    logging.info("Successfully saved model state_dict.")
    with open(params_fp, "wb") as fp:
        pickle.dump(params, fp)
    logging.info("Successfully saved params file.")
