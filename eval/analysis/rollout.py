import torch
import numpy as np
from analysis.io_utils import save_numpy_data

def n_step_rollout(model, ic, n=1, train_tendencies=False):
    """Produce an n-step forward roll-out
    prediction.
    Args:
        model: trained pytorch model.
        ic: [B=1, C, T, X, Y] initial condition for prediction.
        n (int): number of steps to predict for.
    Returns:
        pred: [B=n, C, T, X, Y] n-step model prediction (time along dim=0)."""

    pred = []
    with torch.no_grad():
        idx = torch.tensor([0])
        if train_tendencies:
            # WARNING: if num_out_frames > 1, only top frame is kept in auto-regressive rollout
            if ic.shape[2] > 1:
                for i in range(n):
                    # Use index_select to prevent reducing along dim of size 1
                    pred_temp = torch.index_select(ic, 2, index=idx) + torch.index_select(model(ic), 2, index=idx)
                    pred.append(pred_temp)
                    ic = torch.cat([pred_temp, ic[:,:,:-1,:,:]], dim=2)
            else:
                for i in range(n):
                    pred_temp = ic + torch.index_select(model(ic), 2, index=idx)
                    pred.append(pred_temp)
                    ic = pred_temp
        else:
            if ic.shape[2] > 1:
                for i in range(n):
                    pred_temp = torch.index_select(model(ic), 2, index=idx)
                    pred.append(pred_temp)          
                    ic = torch.cat([pred_temp, ic[:,:,:-1,:,:]], dim=2)
            else:
                for i in range(n):
                    pred_temp = torch.index_select(model(ic), 2, index=idx)
                    pred.append(pred_temp)
                    ic = pred_temp

    pred = torch.cat(pred, dim=0)

    return pred

# def n_step_rollout(model, ic, n=1, save_data=False, train_tendencies=False, save_dir=None):
#     """
#     Produce an n-step forward roll-out prediction.
    
#     Args:
#         model: trained pytorch model.
#         ic (torch.Tensor): [B=1, C, T, X, Y] initial condition for prediction.
#         n (int): number of steps to predict for.
#         save_data (bool): if True, save each predicted step.
#         train_tendencies (bool): indicates model training mode.
#         save_dir (str): directory to save data if save_data = True

#     Returns:
#         pred (torch.Tensor): [B=n, C, T, X, Y] n-step model prediction (time along dim=0).
#     """
#     pred = []
#     with torch.no_grad():
#         idx = torch.tensor([0])
#         if train_tendencies:
#             if ic.shape[2] > 1:
#                 for i in range(n):
#                     pred_temp = ic[...,0:1,:,:] + model(ic)[...,0:1,:,:]
#                     if save_data and save_dir is not None:
#                         data_path = f'{save_dir}/pred_{i}.npy'
#                         save_numpy_data(data_path, pred_temp.squeeze().detach().cpu().numpy())
#                     pred.append(pred_temp)
#                     ic = torch.cat([pred_temp, ic[...,:-1,:,:]], dim=2)
#             else:
#                 for i in range(n):
#                     pred_temp = ic + model(ic)[...,0:1,:,:]
#                     if save_data and save_dir is not None:
#                         data_path = f'{save_dir}/pred_{i}.npy'
#                         save_numpy_data(data_path, pred_temp.squeeze().detach().cpu().numpy())
#                     pred.append(pred_temp)
#                     ic = pred_temp
#         else:
#             if ic.shape[2] > 1:
#                 for i in range(n):
#                     pred_temp = model(ic)[...,0:1,:,:]
#                     if save_data and save_dir is not None:
#                         data_path = f'{save_dir}/pred_{i}.npy'
#                         save_numpy_data(data_path, pred_temp.squeeze().detach().cpu().numpy())
#                     pred.append(pred_temp)
#                     ic = torch.cat([pred_temp, ic[...,:-1,:,:]], dim=2)
#             else:
#                 for i in range(n):
#                     pred_temp = model(ic)[...,0:1,:,:]
#                     if save_data and save_dir is not None:
#                         data_path = f'{save_dir}/pred_{i}.npy'
#                         save_numpy_data(data_path, pred_temp.squeeze().detach().cpu().numpy())
#                     pred.append(pred_temp)
#                     ic = pred_temp

#     pred = torch.cat(pred, dim=0)
#     return pred
