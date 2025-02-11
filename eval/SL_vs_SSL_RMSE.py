import sys
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

import analyze_model


def main(config, ckpt_root, run_num, ckpts, leadtimes):
    """
    Main function to perform SL vs SSL analysis.
    """
    
    # SL performance
    results_SL = analyze_model.main(config)

    # SSL performance
    results_SSL = {}
    config["dataset_params"]["root_dir"] = ckpt_root
    config["dataset_params"]["run_num"] = run_num

    for ckpt in ckpts:
        
        # Set model filename in config
        config["dataset_params"]["model_filename"] = f'training_checkpoints/ckpt_{ckpt}.tar'
    
        results_SSL[ckpt] = analyze_model.main(config)

    # SL vs SSL comparison
    fig, axs = plt.subplots(nrows=3, ncols=3,
                            sharex=True, sharey=True)
    axs = axs.flatten()

    font = {'size': 14}
    mpl.rc('font', **font)

    for i, lead in enumerate(leadtimes):
        # Plot SL
        rmse_median_SL = results_SL['rmse_u_median'][lead] * np.array(len(ckpts))
        rmse_uq_SL = results_SL['rmse_u_uq'][lead] * np.array(len(ckpts))
        rmse_lq_SL = results_SL['rmse_u_lq'][lead] * np.array(len(ckpts))
        axs[i].plot(ckpts, rmse_median_SL, '-k')
        axs[i].fill_between(ckpts, rmse_lq_SL, rmse_uq_SL, color='k', alpha=0.1)

        # Plot SSL
        rmse_median_SSL = [results_SSL[ckpt]['rmse_u_median'][lead] for ckpt in ckpts]
        rmse_uq_SSL = [results_SSL[ckpt]['rmse_u_uq'][lead] for ckpt in ckpts]
        rmse_lq_SSL = [results_SSL[ckpt]['rmse_u_lq'][lead] for ckpt in ckpts]
        axs[i].boxplot(
                [rmse_lq_SSL, rmse_median_SSL, rmse_uq_SSL],
                positions=ckpts,
                vert=True,
                patch_artist=True,
                showmeans=False,
                notch=True
                )

        axs[i].set_xlabel('Number of finetuing epochs')
        axs[i].set_ylabel('RMSE')
        axs[i].grid(True)
        axs[i].set_xlim([ckpts[0], ckpts[-1]])
        axs[i].set_ylim([0, 3.5])
        axs[i].text(0, -1, r'$\Delta t$)', transform=axs[i].transAxes, fontsize=12, 
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    fig.set_tight_layout(True)
    fig.savefig('SL_vs_SSL_RMSE_U.svg')


if __name__ == "__main__":
    
    # Load configuration file for SL
    
    config_fp = str(sys.argv[1])
    with open("config/" + config_filename, "r") as f:
        config = yaml.safe_load(f)

    # Checkpoint parameters for SSL (assuming same config as for SL)

    ckpt_root = str(sys.argv[2])   # Directory for all model checkpoints
    run_num = str(sys.argv[3])   # Run num for SSL

    # List analysis parameters
    ckpts = [1, 25]   # List of epochs-checkpoints
    leadtimes = [0, 24, 49, 74, 99, 124]   # Prediction lead times for which to plot RMSE


    main(config, ckpt_root, run_num, ckpts, leadtimes)
