import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

### Plot imports and settings start here
import seaborn as sns
sns.set_context('poster', font_scale=2.5)
import matplotlib as mpl


mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'cmu serif'
mpl.rcParams.update({'font.size': 22})
# Axis names of the plots
plot_dict = {
    'UDC': r'$U_{\mathrm{DC}}\,\mathrm{/\,V}$',
    'Ld': r'$L_{\mathrm{d}}\,\mathrm{/\,H}$',
    'Lq': r'$L_{\mathrm{q}}\,\mathrm{/\,H}$',
    'Rs': r'$R_{\mathrm{s}}\,\mathrm{/}\,\Omega$',
    'p': r'$p$',
    'Psip': r'$\varPsi_{\mathrm{p}}\,\mathrm{/\,Vs}$',
    'In': r'$I_{\mathrm{n}}\,\mathrm{/\,A}$',
    'Omegan': r'$\omega\,\mathrm{/\,s^{-1}}$',
    'p1': r'$|p_{1}|$',
    'p2': r'$|p_{2}|$',
    'p3': r'$|p_{3}|$',
    'p4': r'$|p_{4}|$',
    'p5': r'$|p_{5}|$',
    'p6': r'$|p_{6}|$',
    'p7': r'$|p_{7}|$',
}
# Whether the plot axis should be logarithmic
log_dict = {
    'UDC': False,
    'Ld': False,
    'Lq': False,
    'Rs': True,
    'p': False,
    'Psip': False,
    'In': True,
    'Omegan': False,
    'p1': True,
    'p2': True,
    'p3': True,
    'p4': True,
    'p5': True,
    'p6': True,
    'p7': True,

}
### Plot imports and settings end here


def plot_comparison(minor_db1, major_db1, minor_db2, major_db2, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    columns = list(minor_db1.columns)
    minor_db1.reset_index(inplace=True)

    plot_combinations = []
    for i in range(len(columns)):
        for j in range(i+1,len(columns)):
            plot_combinations.append([columns[i], columns[j]])

    for combination in plot_combinations:
        fig,axs = plt.subplots(2,2, sharex=True, sharey=True)
        fig.set_size_inches(20,20)
        axs[0][0].plot(np.abs(minor_db1[14:][combination[0]].to_numpy()), np.abs(minor_db1[14:][combination[1]].to_numpy()), 'o',  label='Synthetic', color='orange')
        axs[0][0].plot(np.abs(minor_db1[:14][combination[0]].to_numpy()), np.abs(minor_db1[:14][combination[1]].to_numpy()), 'o', label='Real')
        axs[0][1].plot(np.abs(major_db1[combination[0]].to_numpy()), np.abs(major_db1[combination[1]].to_numpy()), 'o',label = 'Real')
        axs[0][0].set_ylabel(plot_dict[combination[1]], labelpad=10)
        axs[0][0].set_title('IPMSMs (Training)', pad=20)
        axs[0][1].set_title('SPMSMs (Training)', pad=20)

        axs[1][0].plot(np.abs(minor_db2[combination[0]].to_numpy()),
                       np.abs(minor_db2[combination[1]].to_numpy()), 'o', label='Synthetic', color='orange')
        axs[1][1].plot(np.abs(major_db2[combination[0]].to_numpy()), np.abs(major_db2[combination[1]].to_numpy()), 'o', label = 'Real')
        axs[1][0].set_ylabel(plot_dict[combination[1]], labelpad=10)
        axs[1][0].set_title('IPMSMs (Test)',pad=20)
        axs[1][1].set_title('SPMSMs (Test)',pad=20)

        for ax in axs[0]:
            if log_dict[combination[1]]:
                ax.set_yscale('log')
            if log_dict[combination[0]]:
                ax.set_xscale('log')
            ax.tick_params(direction='in', which='both', pad=10)
            ax.set_xlabel(plot_dict[combination[0]], labelpad=10)
            ax.legend()
            ax.grid()
        for ax in axs[1]:
            if log_dict[combination[1]]:
                ax.set_yscale('log')
            if log_dict[combination[0]]:
                ax.set_xscale('log')
            ax.tick_params(direction='in', which='both', pad=10)
            ax.set_xlabel(plot_dict[combination[0]], labelpad=10)
            ax.legend()
            ax.grid()
        fig.tight_layout()
        file = f"{combination[0]}_{combination[1]}.png"
        plt.savefig(save_path / file, format='png')
        plt.close(fig)

#Import databases
code_path = Path(__file__).parent.absolute()
motor_db_path = code_path.parent.parent / "MotorDB"
ode_train_db = pd.read_excel(motor_db_path / "ODETraining.xlsx")
ode_test_db = pd.read_excel(motor_db_path / "ODETest.xlsx")
train_db = pd.read_excel(motor_db_path / "Training.xlsx")
test_db = pd.read_excel(motor_db_path / "Test.xlsx")

#Find IPMSMs, SPMSMs and the original 14 IPMSMs
train_ipmsm = train_db[train_db['Ld'] != train_db['Lq']]
fake_ipmsm = train_ipmsm[(train_ipmsm['Rs'] == 1) & (train_ipmsm['p'] == 4) & (train_ipmsm['In'] == 5)]
train_ipmsm.drop(fake_ipmsm.index, inplace=True)
train_ipmsm = pd.concat([train_ipmsm,fake_ipmsm])
train_ipmsm_ode = ode_train_db.loc[train_ipmsm.index]
train_ipmsm.reset_index(inplace=True, drop=True)

test_ipmsm = test_db[test_db['Ld'] != test_db['Lq']]
test_ipmsm_ode = ode_test_db.loc[test_ipmsm.index]

train_spmsm = train_db[train_db['Ld'] == train_db['Lq']]
train_spmsm_ode = ode_train_db.loc[train_spmsm.index]
test_spmsm = test_db[test_db['Ld'] == test_db['Lq']]
test_spmsm_ode = ode_test_db.loc[test_spmsm.index]

#Save pictures in MotorDB folder
save_path = code_path.parent.parent / "MotorDB" / "Visualization"
plot_comparison(train_ipmsm, train_spmsm, test_ipmsm, test_spmsm, save_path)
plot_comparison(train_ipmsm_ode, train_spmsm_ode, test_ipmsm_ode, test_spmsm_ode, save_path)