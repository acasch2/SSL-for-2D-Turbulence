import sys
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from ruamel.yaml import YAML
import yaml

import analyze_model


def main(config, ckpt_root, run_num, ckpts, leadtimes):
    """
    Main function to perform SL vs SSL analysis.
    """
    
    # SL performance
    print(f'\nSL analysis ...')

    results_SL = analyze_model.main(config)

    # SSL performance
    print(f'\nSSL analysis ...')

    results_SSL = {}
    config["dataset_params"]["root_dir"] = ckpt_root
    config["dataset_params"]["run_num"] = run_num

    for ckpt in ckpts:
        print(f'On epoch-ckpt: {ckpt}')

        # Set model filename in config
        config["dataset_params"]["model_filename"] = f'training_checkpoints/ckpt_{ckpt}.tar'
    
        results_SSL[ckpt] = analyze_model.main(config)

    # SL vs SSL comparison
    print(f'Plotting results ...')

    fig, axs = plt.subplots(nrows=2, ncols=2,
                            sharex=True, sharey=True)
    axs = axs.flatten()

    font = {'size': 14}
    mpl.rc('font', **font)

    for i, lead in enumerate(leadtimes):
        # Plot SL
        rmse_median_SL = results_SL['rmse_u_median'][lead] * np.ones(len(ckpts))
        rmse_uq_SL = results_SL['rmse_u_uq'][lead] * np.ones(len(ckpts))
        rmse_lq_SL = results_SL['rmse_u_lq'][lead] * np.ones(len(ckpts))
        axs[i].plot(ckpts, rmse_median_SL, '-k')
        axs[i].fill_between(ckpts, rmse_lq_SL, rmse_uq_SL, color='k', alpha=0.1)

        # Plot SSL
        rmse_median_SSL = [results_SSL[ckpt]['rmse_u_median'][lead] for ckpt in ckpts]
        rmse_uq_SSL = [results_SSL[ckpt]['rmse_u_uq'][lead] for ckpt in ckpts]
        rmse_lq_SSL = [results_SSL[ckpt]['rmse_u_lq'][lead] for ckpt in ckpts]
        rmse_SSL = [[rmse_lq_SSL[j], rmse_median_SSL[j], rmse_uq_SSL[j]] for j in range(len(ckpts))]
        box_widths = 0.15 * ((ckpts[-1] - ckpts[0]) / len(ckpts))
        axs[i].boxplot(
                rmse_SSL,
                positions=ckpts,
                vert=True,
                patch_artist=True,
                showmeans=False,
                notch=False,
                widths=box_widths
                )

        #axs[i].set_xlabel('Number of finetuing epochs')
        #axs[i].set_ylabel('RMSE')
        axs[i].grid(True)
        x_pad = (0.2 / 0.15) * box_widths
        axs[i].set_xlim([ckpts[0]-x_pad, ckpts[-1]+x_pad])
        axs[i].set_ylim([-0.25, 3.5])
        axs[i].set_title(f'{lead+1} {r"$\Delta t$"}', fontsize=8)
        axs[i].tick_params(axis='x', labelrotation=45)
        #axs[i].text(0, -1, f'{lead} {r"$\Delta t$"}', fontsize=10, 
        #            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
   
    fig.supxlabel('Number of finetuning epochs')
    fig.supylabel('RMSE')
    fig.set_tight_layout(True)
    fig.savefig(f'SL_vs_SSL_RMSE_U_{run_num}.svg')


if __name__ == "__main__":
    
    # Load configuration file for SL
    
    config_fp = str(sys.argv[1])
    with open("config/" + config_fp, "r") as f:
        config = yaml.safe_load(f)

    # Checkpoint parameters for SSL (assuming same config as for SL)

    ckpt_root = str(sys.argv[2])   # Directory for all model checkpoints
    run_num = str(sys.argv[3])   # Run num for SSL

    # List analysis parameters
    ckpts = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]   # List of epochs-checkpoints
    leadtimes = [0, 32, 65, 99]   # Prediction lead times for which to plot RMSE


    main(config, ckpt_root, run_num, ckpts, leadtimes)
