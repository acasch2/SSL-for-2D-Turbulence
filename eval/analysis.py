import sys
import os
import logging

import torch
import torch.distributed as dist
from src.models.vision_transformer import ViT
from src.utils.data_utils import get_dataloader
from src.utils.io_utils import load_params
from src.analysis.short_analysis import perform_short_analysis
from src.analysis.long_analysis import perform_long_analysis
from src.utils.visualization import plot_analysis

logging.basicConfig(level=logging.INFO)

def main(root_dir, model_filename, params_filename, test_length, num_tests, test_file_start_idx,
         analysis_dict, test_length_climo=10000, test_length_video=500, run_num='base'):
    """
    Main function to run analysis on trained model.
    """
    params_fp = os.path.join(root_dir, params_filename)
    params = load_params(params_fp)
    logging.info(f'params: {params}')

    # Initiate model and load weights
    model_fp = os.path.join(root_dir, model_filename)
    model = ViT(
        img_size=params["img_size"],
        patch_size=params["patch_size"],
        num_frames=params["num_frames"],
        tubelet_size=params["tubelet_size"],
        in_chans=params["in_chans"],
        encoder_embed_dim=params["encoder_embed_dim"],
        encoder_depth=params["encoder_depth"],
        encoder_num_heads=params["encoder_num_heads"],
        decoder_embed_dim=params["decoder_embed_dim"],
        decoder_depth=params["decoder_depth"],
        decoder_num_heads=params["decoder_num_heads"],
        mlp_ratio=params["mlp_ratio"],
        num_out_frames=params["num_out_frames"],
        patch_recovery=params["patch_recovery"]
        )

    ckpt_temp = torch.load(model_fp, map_location=torch.device('cpu'))['model_state']
    ckpt = {key[7:]: val for key, val in ckpt_temp.items()}
    model.load_state_dict(ckpt)
    model.eval()

    # Initiate dataloaders
    test_file_range = (test_file_start_idx, test_file_start_idx+(test_length*num_tests*params["target_step"])-1)
    dataloader, dataset = get_dataloader(data_dir=params["data_dir"],
                                         file_range=test_file_range,
                                         target_step=params["target_step"],
                                         train_tendencies=params["train_tendencies"],
                                         batch_size=test_length,
                                         train=False,
                                         stride=params["target_step"],
                                         distributed=dist.is_initialized(),
                                         num_frames=params["num_frames"],
                                         num_out_frames=params["num_out_frames"],
                                         num_workers=2,
                                         pin_memory=params["pin_memory"])

    test_file_range_climo = (test_file_start_idx, test_file_start_idx+test_length_climo)
    dataloader_climo, dataset_climo = get_dataloader(data_dir=params["data_dir"],
                                                     file_range=test_file_range_climo,
                                                     target_step=1,
                                                     train_tendencies=params["train_tendencies"],
                                                     batch_size=test_length_climo,
                                                     train=False,
                                                     stride=1,
                                                     distributed=dist.is_initialized(),
                                                     num_frames=1,
                                                     num_out_frames=1,
                                                     num_workers=2,
                                                     pin_memory=params["pin_memory"])

    test_file_range_video = (test_file_start_idx, test_file_start_idx+(test_length_video*params["target_step"])+params["target_step"])
    dataloader_video, dataset_video = get_dataloader(data_dir=params["data_dir"],
                                                     file_range=test_file_range_video,
                                                     target_step=params["target_step"],
                                                     train_tendencies=params["train_tendencies"],
                                                     batch_size=test_length_video,
                                                     train=False,
                                                     stride=params["target_step"],
                                                     distributed=dist.is_initialized(),
                                                     num_frames=params["num_frames"],
                                                     num_out_frames=params["num_out_frames"],
                                                     num_workers=2,
                                                     pin_memory=params["pin_memory"])

    # Climatology calculation
    climo_data, _ = next(iter(dataloader_climo))
    climo_data = climo_data.transpose(-1, -2).squeeze().detach().cpu().numpy()
    climo_u = climo_data[:,0].mean(axis=0)
    climo_v = climo_data[:,1].mean(axis=0)
    climo_u_zonal = climo_data[:,0].mean(axis=(0,2))

    # Perform short analysis
    results_short = perform_short_analysis(model, dataloader, dataset, climo_u, climo_v, params, analysis_dict)

    # Perform long analysis
    save_dir = os.path.join(root_dir)  # or create a separate directory if desired
    results_long = perform_long_analysis(model, dataloader_video, dataset, climo_u, climo_u_zonal, climo_v, params, analysis_dict, save_dir)

    # Combine results
    results = {**results_short, **results_long}

    # Plot analysis
    plot_analysis(results, analysis_dict, run_num, save_dir)


if __name__ == "__main__":
    # Example args - these would normally be parsed from sys.argv or argparse
    run_num = 'base'
    root_dir = f'/ocean/projects/phy220045p/jakhar/2d_emulator_vision/SSL-for-2D-Turbulence/results/BASE/{run_num}/'
    model_filename = 'training_checkpoints/best_ckpt.tar'
    params_filename = 'hyperparams.yaml'

    test_length_short = 100
    num_tests = 10
    test_file_start_idx = 350000
    test_length_climo = 1000
    test_length_long = 1000
    pca_ncomp = 3

    analysis_dict = {
        'rmse': False,
        'acc': False,
        'spectra': False,
        'spectra_leadtimes': [0, 4, 9, 49, 99],
        'save_data': False,
        'zonal_pca': False,
        'pca_ncomp': pca_ncomp,
        'video': False,
        'div': True,
        'long_rollout_length': test_length_long,
        'EOF_target': False
    }

    main(root_dir, model_filename, params_filename, test_length_short, num_tests, test_file_start_idx,
         analysis_dict, test_length_climo=test_length_climo, test_length_video=test_length_long, run_num=run_num)


# # Test skill of trained model.
# import sys
# import torch
# from torch import nn
# import torch.distributed as dist
# import os
# import ast
# from scipy.io import loadmat
# import numpy as np
# from sklearn.decomposition import PCA
# from matplotlib import pyplot as plt
# from matplotlib import cm
# import matplotlib as mpl
# from ruamel.yaml import YAML
# import imageio

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# src_dir = os.path.join(parent_dir, 'src')
# sys.path.append(parent_dir)
# sys.path.append(src_dir)
# from src.models.vision_transformer import ViT
# from src.utils.data_loaders import get_dataloader

# from utils.metrics import get_rmse, get_acc, spectrum_zonal_average_2D_FHIT, get_spectra, get_zonal_PCA, get_div
# from utils.visualization import plot_analysis, make_video
# from analysis.short_analysis import perform_short_analysis
# from analysis.rollout import n_step_rollout
# from analysis.long_analysis import perform_long_analysis

# # ================================================================================ #

# # def n_step_rollout(model, ic, n=1, save_data=False, train_tendencies=False):
# #     """Produce an n-step forward roll-out
# #     prediction.
# #     Args:
# #         model: trained pytorch model.
# #         ic: [B=1, C, T, X, Y] initial condition for prediction. [Batch, Channel, Timesteps (only for multi-timestep trainng), X, Y]
# #         n (int): number of steps to predict for.
# #     Returns:
# #         pred: [B=n, C, T, X, Y] n-step model prediction (time along dim=0)."""

# #     pred = []
# #     with torch.no_grad():
# #         idx = torch.tensor([0])
# #         if train_tendencies:
# #             # WARNING: if num_out_frames > 1, only top frame is kept in auto-regressive rollout
# #             if ic.shape[2] > 1:
# #                 for i in range(n):
# #                     # Use index_select to prevent reducing along dim of size 1
# #                     pred_temp = torch.index_select(ic, 2, index=idx) + torch.index_select(model(ic), 2, index=idx)
# #                     if save_data:
# #                         np.save(f'pred_{i}.npy', pred_temp.squeeze().detach().cpu().numpy())
# #                     else:
# #                         pred.append(pred_temp)
# #                     pred.append(pred_temp)
# #                     ic = torch.cat([pred_temp, ic[:,:,:-1,:,:]], dim=2)
# #             else:
# #                 for i in range(n):
# #                     pred_temp = ic + torch.index_select(model(ic), 2, index=idx)
# #                     pred.append(pred_temp)
# #                     ic = pred_temp
# #         else:
# #             if ic.shape[2] > 1:
# #                 for i in range(n):
# #                     pred_temp = torch.index_select(model(ic), 2, index=idx)
# #                     pred.append(pred_temp)          
# #                     ic = torch.cat([pred_temp, ic[:,:,:-1,:,:]], dim=2)
# #             else:
# #                 for i in range(n):
# #                     pred_temp = torch.index_select(model(ic), 2, index=idx)
# #                     pred.append(pred_temp)
# #                     ic = pred_temp

# #     pred = torch.cat(pred, dim=0)

# #     return pred

# # def perform_analysis(model, dataloader, dataloader_climo, dataloader_video, dataset, analysis_dict, params):
# #     """
# #     Returns:
# #         results: dictionary of {'rmse','rmse_per','acc','acc_per','spectra'}
# #     """

# #     # Calculate climatology
# #     climo_data, _ = next(iter(dataloader_climo))
# #     print(f'climo_data.shape: {climo_data.shape}')

# #     climo_data = climo_data.transpose(-1, -2).squeeze().detach().cpu().numpy()                     # [B=n_steps, C, X, Y]
# #     climo_u = climo_data[:,0].mean(axis=0)                                       # [X, Y]
# #     climo_v = climo_data[:,1].mean(axis=0) 
# #     climo_u_zonal = climo_data[:,0].mean(axis=(0,2))                                  
# #     print(f'climo_u.shape: {climo_u.shape}')      
# #     print(f'climo_u_zonal.shape: {climo_u_zonal.shape}')

# #     results = {}

# #     short_analysis = any([analysis_dict['rmse'], analysis_dict['acc'], analysis_dict['spectra']])
# #     if short_analysis:
# #         print('************ Short analysis ************')
# #                         # should be [X, Y]

# #         rmse_u, rmse_u_per = [], []
# #         rmse_v, rmse_v_per = [], []
# #         acc_u, acc_u_per = [], []
# #         acc_v, acc_v_per = [], []
# #         spectra, spectra_tar, wavenumbers = [], [], []
# #         for i, batch in enumerate(dataloader):

# #             print(f'Prediction iteration: {i}\n')
# #             inputs, targets = batch
# #             ic = inputs[0].unsqueeze(dim=0)
# #             print(f'ic.shape: {ic.shape}')
# #             print(inputs.shape)
# #             n_steps = inputs.shape[0]

# #             pred = n_step_rollout(model, ic, n=n_steps, train_tendencies=params["train_tendencies"])
# #             per_pred = inputs[0].repeat(n_steps, 1, 1, 1, 1)[:,:,0,:,:]

# #             print(f'pred.shape: {pred.shape}')                                      # should be: [B=n_steps, C, T=1, X, Y]
# #             print(f'per_pred.shape: {per_pred.shape}')
# #             print(f'targets.shape: {targets.shape}')

# #             pred = pred.transpose(-1,-2).squeeze().detach().cpu().numpy()                            # [B=n_steps, C, X, Y]
# #             per_pred = per_pred.transpose(-1,-2).squeeze().detach().cpu().numpy()  
# #             targets = targets.transpose(-1,-2).squeeze().detach().cpu().numpy()

# #             pred_u = pred[:,0]                                                      # [B=n_steps, X, Y]
# #             pred_v = pred[:,1]
# #             per_pred_u = per_pred[:,0]
# #             per_pred_v = per_pred[:,1]
# #             tar_u = targets[:,0]
# #             tar_v = targets[:,1]

# #             # Unnormalize data
# #             pred_u = (pred_u *  dataset.input_std[0] + dataset.input_mean[0]) 
# #             pred_v = (pred_v * dataset.input_std[1] + dataset.input_mean[1]) 
# #             tar_u = (tar_u * dataset.label_std[0] + dataset.label_mean[0]) 
# #             tar_v = (tar_v * dataset.label_std[1] + dataset.label_mean[1])

# #             if analysis_dict['rmse']:
# #                 rmse_u.append(get_rmse(tar_u, pred_u, climo=climo_u))
# #                 rmse_u_per.append(get_rmse(tar_u, per_pred_u, climo=climo_u))
# #                 rmse_v.append(get_rmse(tar_v, pred_v, climo=climo_v))
# #                 rmse_v_per.append(get_rmse(tar_v, per_pred_v, climo=climo_v))
# #                 print(f'rmse_u[-1].shape: {rmse_u[-1].shape}\n')

# #             if analysis_dict['acc']:
# #                 acc_u.append(get_acc(tar_u, pred_u, climo_u))                       # [B=n_steps,]
# #                 acc_u_per.append(get_acc(tar_u, per_pred_u, climo_u))
# #                 acc_v.append(get_acc(tar_v, pred_v, climo_v))
# #                 acc_v_per.append(get_acc(tar_v, per_pred_v, climo_v))
# #                 print(f'acc_u[-1].shape: {acc_u[-1].shape}\n')

# #             if analysis_dict['spectra']:
# #                 spectra_temp, wavenumbers = get_spectra(pred_u, pred_v)             # [B=n_steps, k]
# #                 spectra_tar_temp, _ = get_spectra(tar_u, tar_v)
# #                 spectra.append(spectra_temp)
# #                 spectra_tar.append(spectra_tar_temp)
# #                 print(f'spectra[-1].shape: {spectra[-1].shape}\n')

# #         # Average over all predictions and Save results
# #         if analysis_dict['rmse']:
# #             results['rmse_u_mean'] = np.mean(np.stack(rmse_u, axis=0), axis=0)
# #             results['rmse_u_std'] = np.std(np.stack(rmse_u, axis=0), axis=0)
# #             results['rmse_u_per_mean'] = np.mean(np.stack(rmse_u_per, axis=0), axis=0)
# #             results['rmse_u_per_std'] = np.std(np.stack(rmse_u_per, axis=0), axis=0)
# #             results['rmse_v_mean'] = np.mean(np.stack(rmse_v, axis=0), axis=0)
# #             results['rmse_v_std'] = np.std(np.stack(rmse_v, axis=0), axis=0)
# #             results['rmse_v_per_mean'] = np.mean(np.stack(rmse_v_per, axis=0), axis=0)
# #             results['rmse_v_per_std'] = np.std(np.stack(rmse_v_per, axis=0), axis=0)
# #         if analysis_dict['acc']:
# #             results['acc_u_mean'] = np.mean(np.stack(acc_u, axis=0), axis=0)
# #             results['acc_u_std'] = np.std(np.stack(acc_u, axis=0), axis=0)
# #             results['acc_u_per_mean'] = np.mean(np.stack(acc_u_per, axis=0), axis=0)
# #             results['acc_u_per_std'] = np.std(np.stack(acc_u_per, axis=0), axis=0)
# #             results['acc_v_mean'] = np.mean(np.stack(acc_v, axis=0), axis=0)
# #             results['acc_v_std'] = np.std(np.stack(acc_v, axis=0), axis=0)
# #             results['acc_v_per_mean'] = np.mean(np.stack(acc_v_per, axis=0), axis=0)
# #             results['acc_v_per_std'] = np.std(np.stack(acc_v_per, axis=0), axis=0)
# #         if analysis_dict['spectra']:
# #             results['spectra'] = np.mean(np.stack(spectra, axis=0), axis=0)          # [B=n_steps, k]
# #             results['spectra_tar'] = np.mean(np.stack(spectra_tar, axis=0), axis=0)
# #             results['wavenumbers'] = wavenumbers


# #     long_analysis = any([analysis_dict['video'], analysis_dict['zonal_pca'], analysis_dict['div']])
# #     if long_analysis:
# #         print('************ Long analysis ************')

# #         inp, tar = next(iter(dataloader_video))
# #         ic = inp[0].unsqueeze(dim=0)
# #         n_steps = inp.shape[0]

# #         pred = n_step_rollout(model, ic, n=n_steps, train_tendencies=params["train_tendencies"])
# #         #pred = pred[1:]

# #         pred = pred.transpose(-1,-2).squeeze().detach().cpu().numpy()
# #         tar = tar.transpose(-1,-2).squeeze().detach().cpu().numpy()

# #         print(pred.shape)

# #         # Unnormalize data
# #         pred[:,0] = (pred[:,0]  * dataset.input_std[0] + dataset.input_mean[0])
# #         pred[:,1] = (pred[:,1] * dataset.input_std[1] + dataset.input_mean[1])
# #         tar[:,0] = (tar[:,0] * dataset.label_std[0] + dataset.label_mean[0])
# #         tar[:,1] = (tar[:,1] * dataset.label_std[1] + dataset.label_mean[1]) 

# #         print('IC shape:', ic.shape)
# #         print('pred shape:', pred.shape)
# #         print('tar shape:', tar.shape)


# #         pred_u = pred[:,0]
# #         pred_v = pred[:,1]
# #         tar_u = tar[:,0]
# #         tar_v = tar[:,1]

# #         print(f'pred_u.shape: {pred_u.shape}\n')
# #         print(f'tar_u.shape: {tar_u.shape}\n')
# #         print(f'pred_v.shape: {pred_v.shape}\n')
# #         print(f'tar_v.shape: {tar_v.shape}\n')

# #         if analysis_dict['video']:
# #             print(f'Making long roll-out video.')
# #             make_video(pred, tar)

# #         if analysis_dict['zonal_pca']:
# #             print('******* U shape *****', pred_u.shape, tar_u.shape) # [B=n_steps, X, Y]

# #             pred_u_zonal = np.mean(pred_u, axis=-1)
# #             pred_u_anom = pred_u_zonal - climo_u_zonal
# #             pred_u_pc, pred_u_eof = get_zonal_PCA(pred_u_anom,
# #                                                     n_comp=analysis_dict['pca_ncomp'])
            
# #             tar_u_zonal = np.mean(tar_u, axis=-1)
# #             tar_u_anom = tar_u_zonal - climo_u_zonal
# #             tar_u_pc, tar_u_eof = get_zonal_PCA(tar_u_anom,
# #                                                     n_comp=analysis_dict['pca_ncomp'])
# #             print(f'pred_u_pc.shape: {pred_u_pc.shape}\n')
# #             print(f'pred_u_eof.shape: {pred_u_eof.shape}\n')
            
# #             results['pred_u_pc'] = pred_u_pc
# #             results['pred_u_eof'] = pred_u_eof
# #             results['tar_u_pc'] = tar_u_pc
# #             results['tar_u_eof'] = tar_u_eof

# #         if analysis_dict['div']:
# #             print(f'div. tar_u.shape, tar_v.shape: {tar_u.shape}, {tar_v.shape}\n')
# #             print(f'div. pred_u.shape, pred_v.shape: {pred_u.shape}. {pred_v.shape}\n')

# #             # Rescale preds/tars for div calculation
# #             pred_div = get_div(pred_u, pred_v)
# #             tar_div = get_div(tar_u, tar_v)

# #             results['pred_div'] = pred_div
# #             results['tar_div'] = tar_div

# #     return results

# def main(root_dir, model_filename, params_filename, test_length, num_tests, test_file_start_idx,
#         analysis_dict, test_length_climo=10000, test_length_video=500):

#     # Read in params file
#     params_fp = os.path.join(root_dir, params_filename)
#     yaml = YAML(typ='safe')
#     with open(params_fp, 'r') as f:
#         params_temp = yaml.load(f)
    
#     params = {}
#     for key, val in params_temp.items():
#         try:
#             params[key] = ast.literal_eval(val)
#         except:
#             params[key] = val
#     print(f'params: {params}\n')

#     # Initiate model and load weights
#     model_fp = os.path.join(root_dir, model_filename)
#     model = ViT(
#         img_size=params["img_size"],
#         patch_size=params["patch_size"],
#         num_frames=params["num_frames"],
#         tubelet_size=params["tubelet_size"],
#         in_chans=params["in_chans"],
#         encoder_embed_dim=params["encoder_embed_dim"],
#         encoder_depth=params["encoder_depth"],
#         encoder_num_heads=params["encoder_num_heads"],
#         decoder_embed_dim=params["decoder_embed_dim"],
#         decoder_depth=params["decoder_depth"],
#         decoder_num_heads=params["decoder_num_heads"],
#         mlp_ratio=params["mlp_ratio"],
#         num_out_frames=params["num_out_frames"],
#         patch_recovery=params["patch_recovery"]
#         )
#     ckpt_temp = torch.load(model_fp, map_location=torch.device('cpu'))['model_state']
#     ckpt = {}
#     for key, val in ckpt_temp.items():
#         key_new = key[7:]
#         ckpt[key_new] = val
    
#     model.load_state_dict(ckpt)
#     model.eval()

#     # Initiate dataloaders
#     test_file_range = (test_file_start_idx, test_file_start_idx+(test_length*num_tests*params["target_step"])-1) #+params["target_step"])
#     dataloader, dataset = get_dataloader(data_dir=params["data_dir"],
#                                     file_range=test_file_range,
#                                     target_step=params["target_step"],
#                                     train_tendencies=params["train_tendencies"],
#                                     batch_size=test_length,
#                                     train=False,
#                                     stride=params["target_step"],
#                                     distributed=dist.is_initialized(),
#                                     num_frames=params["num_frames"],
#                                     num_out_frames=params["num_out_frames"],
#                                     num_workers=2,
#                                     pin_memory=params["pin_memory"])
#     test_file_range = (test_file_start_idx, test_file_start_idx+test_length_climo)
#     dataloader_climo, dataset_climo = get_dataloader(data_dir=params["data_dir"],
#                                     file_range=test_file_range,
#                                     target_step=1,
#                                     train_tendencies=params["train_tendencies"],
#                                     batch_size=test_length_climo,
#                                     train=False,
#                                     stride=1,
#                                     distributed=dist.is_initialized(),
#                                     num_frames=1, #params["num_frames"],
#                                     num_out_frames=1, #params["num_out_frames"],
#                                     num_workers=2,
#                                     pin_memory=params["pin_memory"])
#     test_file_range = (test_file_start_idx, test_file_start_idx+(test_length_video*params["target_step"])+params["target_step"])
#     dataloader_video, datasaet_video = get_dataloader(data_dir=params["data_dir"],
#                                     file_range=test_file_range,
#                                     target_step=params["target_step"],
#                                     train_tendencies=params["train_tendencies"],
#                                     batch_size=test_length_video,
#                                     train=False,
#                                     stride=params["target_step"],
#                                     distributed=dist.is_initialized(),
#                                     num_frames=params["num_frames"],
#                                     num_out_frames=params["num_out_frames"],
#                                     num_workers=2,
#                                     pin_memory=params["pin_memory"])

#     print(f'len(dataset): {len(dataset)}')
#     print(f'len(dataset_climo): {len(dataset_climo)}')
#     print(f'len(dataset_video): {len(datasaet_video)}')

#     results = perform_analysis(model, dataloader, dataloader_climo, dataloader_video, dataset, analysis_dict, params)

#     plot_analysis(results, analysis_dict)


# # ================================================================================ #

# # File Paths
# # root_dir = '/scratch/user/u.dp200518/SSL-2DTurb/BASE/current_best/'
# # model_filename = 'training_checkpoints/best_ckpt.tar'
# # params_filename = 'hyperparams.yaml'
# # run_num = 'current_best'

# run_num = 'base'

# # run_num = sys.argv[1]
# root_dir = '/ocean/projects/phy220045p/jakhar/2d_emulator_vision/SSL-for-2D-Turbulence/results/BASE/' + run_num + '/'
# model_filename = 'training_checkpoints/best_ckpt.tar'
# params_filename = 'hyperparams.yaml'

# # # Test Parameters
# # test_length = 100    # will be batch size
# # num_tests = 5
# # test_file_start_idx = 350000 

# # test_length_climo = 10000

# # test_length_video = 500

# # pca_ncomp = 2


# test_length_short = 100  # Length of the short tests rmse, acc, spectra
# num_tests = 2
# test_file_start_idx = 350000 # Starting index of test files used as initial conditions
# test_length_climo = 10 # Number of snashots to be used to calculate climatology

# long_rollout_length = 10 # Length of long tests EOF, Divergence, Video; 
# video_rollout_length = 5  # Length of the video to be generated

# # Number of EOF's to be calculated
# pca_ncomp = 2


# # Analysis
# analysis_dict = {
#     ### These are short length statistics
#     # Emulator will generate data every time on the fly
#         'rmse': True,
#         'acc': True,
#         'spectra': True,
#         'spectra_leadtimes': [0, 4, 9, 49, 99],
#         # 'spectra_leadtimes': [0, 1, 2, 5],
    
#     ### These are long length statistics
#     # Emulator will generate and save data for the entire length and then perform analysis
#     # If save_data: True, then data will be saved in the results folder # Recommended 
#     # if save_data: False, then data will not be saved and analysis will be performed on the fly - Not recommended will run out of memory
#         'save_data': True,
#         'zonal_pca': True,
#         'pca_ncomp': pca_ncomp,
#         'video': True,
#         'video_rollout_length': video_rollout_length,
#         'div': True,
#         'long_rollout_length': test_length_long,
#         'EOF_target': False
#         }

# main(root_dir, model_filename, params_filename, test_length_short, num_tests, test_file_start_idx,
#         analysis_dict, test_length_climo=test_length_climo, test_length_video=test_length_long) 




