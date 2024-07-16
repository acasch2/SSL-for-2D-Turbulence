# Test skill of trained model.
import sys
import torch
from torch import nn
import os
import pickle
from scipy.io import loadmat
import numpy as np
from py2d.initialize import initialize_wavenumbers_rfft2
from py2d.convert import Omega2Psi, Psi2UV
from matplotlib import pyplot as plt
from matplotlib import cm
print(f'Here 1.')

sys.path.append('/jet/home/dpatel9/SSL-Wavelets/')
sys.path.append('/jet/home/dpatel9/SSL-Wavelets/src/')
from src.models.vision_transformer import ViT
from src.utils.data_loaders import get_dataloader


def n_step_prediction(model, ic, n=1):
    """Produce an n-step forward roll-out 
    prediction.
    Args:
        model: trained pytorch model.
        ic: initial condition for prediction.
        n (int): nnumber of steps to predict for.
    Returns:
        pred: n-step model prediction (time along dim=0)."""
    
    pred = [ic]
    with torch.no_grad():
        for i in range(n):
            pred.append(model(pred[-1]))
    
    pred = torch.cat(pred, dim=0)

    return pred

def get_rmse(target, pred, dims_to_reduce=None):
    err = (target - pred) ** 2
    err = err.mean(dim=dims_to_reduce)
    rmse = torch.sqrt(err)
    return rmse

def get_avg_rmse(dataloader, model):

    rmse, rmse_per = [], []
    for i, batch in enumerate(dataloader):
        
        print(f'Iter: {i}')
        inputs, target = batch
        ic = inputs[0].unsqueeze(dim=0)
        n_steps = inputs.shape[0]

        pred = n_step_prediction(model, ic, n_steps)
        pred = pred[1:]
        per_pred = inputs[0].repeat(n_steps, 1, 1, 1, 1)
        print(f'{per_pred.shape}')

        dims_to_reduce = (1, 2, 3, 4)
        rmse.append(get_rmse(target, pred, dims_to_reduce))
        rmse_per.append(get_rmse(target, per_pred, dims_to_reduce))

    rmse = torch.stack(rmse, dim=0)
    rmse = rmse.mean(dim=0).numpy()
    rmse_per = torch.stack(rmse_per, dim=0)
    rmse_per = rmse_per.mean(dim=0).numpy()

    return rmse, rmse_per, pred, target


print(f'Starting Job.\n')

# File paths
root_dir = '/ocean/projects/atm170004p/dpatel9/ML_Weights/Base_Emulator/'
model_filename = 'test9.pt'
params_filename = 'test9_parameters.pkl'

# Test parameters
test_length = 6
num_tests = 3
target_steps = 1
test_file_range = (350000, 350000+(test_length*num_tests)+target_steps-1)

# ----- Read in parameter file as a python dictionary ----- #
params_fp = os.path.join(root_dir, params_filename)
with open(params_fp, 'rb') as f:
    params = pickle.load(f)
print(f'params: {params}\n')

# ----- Load model ----- #
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
    norm_layer=params["norm_layer"],
    num_out_frames=params["num_out_frames"]
)
model.load_state_dict(torch.load(model_fp, map_location=torch.device('cpu')))
model.eval()

# ----- Get test data ----- #
dataloader, dataset = get_dataloader(data_dir=params["data_dir"],
                                    file_range=test_file_range,
                                    target_step=params["target_step"],
                                    batch_size=test_length,
                                    train=False,
                                    num_workers=params["num_workers"],
                                    pin_memory=params["pin_memory"])

# ----- Get RMSE ----- #
rmse, rmse_per, pred, target = get_avg_rmse(dataloader, model)

# ----- Plot results ----- #
# Plot rmse
fig, ax = plt.subplots()
ax.plot(rmse, '-k')
ax.plot(rmse_per, '-g')
fig_fname = 'RMSE' + ' ' + model_filename + '.png'
fig.savefig(fig_fname)

# Plot single inference
nplot = 6
fig, ax = plt.subplots(figsize=[12,2], nrows=2, ncols=nplot, sharey=True)
axs = ax.flatten()
preds = pred.squeeze(dim=2)
tars = target.squeeze(dim=2)
xx, yy = np.meshgrid(np.arange(preds.shape[-2]), np.arange(preds.shape[-1]))
for i in range(nplot):
    axs[i].pcolor(xx, yy, preds[i][0], cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=-5, vmax=5)
for i, step in enumerate(range(nplot, 2*nplot)):
    axs[step].pcolor(xx, yy, tars[i][0], cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=-5, vmax=5)
#fig.colorbar(surf, shrink=0.5, aspect=5)
fig_fname = 'Predictions_' + model_filename + '.png'
fig.savefig(fig_fname)