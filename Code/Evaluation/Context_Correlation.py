import pandas as pd
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib as mpl
import seaborn as sns
sns.set_context('poster', font_scale=2.5)
mpl.rcParams.update({'font.size': 22})
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'cmu serif'


plot_dict = {
    "UDC": r"$U_\mathrm{{DC}}$",
    "p": r"$p$",
    "Omegan": r"$\omega_{{\mathrm{{me}}}}$",
    "Psip": r"$\psi_{{\mathrm{{p}}}}$",
    "In": r"$I_{\mathrm{n}}$",
    "Rs": r"$R_{\mathrm{s}}$",
    "Lq": r"$L_{\mathrm{q}}$",
    "Ld": r"$L_{\mathrm{d}}$",
    'p1': r'$p_{1}$',
    'p2': r'$p_{2}$',
    'p3': r'$p_{3}$',
    'p4': r'$p_{4}$',
    'p5': r'$p_{5}$',
    'p6': r'$p_{6}$',
    'p7': r'$p_{7}$',
}

for i in range(8):
    plot_dict[f"context{i}"] = r"$z_{{{}}}$".format(i+1)


def gen_plot(df_con, df_par, save_name, save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    df_new = df_con.copy()
    for column in df_par.columns.tolist():
        df_new = pd.concat([df_new, df_par[column]], axis=1)
    corr = df_new.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(cax, orientation="horizontal", pad = 0.05)
    cbar.set_label('Pearson Correlation Coefficient')
    cbar.ax.get_yaxis().labelpad = 30
    ticks = np.arange(0,len(df_new.columns),1)
    ax.set_xticks(ticks)
    ax.tick_params(direction='in', which='both', length=0)
    ax.set_yticks(ticks)
    axislabels = []
    for label in df_new.columns:
        axislabels.append(plot_dict[label])
    ax.set_xticklabels(axislabels)
    ax.set_yticklabels(axislabels)
    ax.tick_params(axis='y', which='major', pad=10)
    fig.set_size_inches(25,29)
    fig.tight_layout()
    plt.savefig(save_path / f"{save_name}.png", format='png', bbox_inches='tight')

parser = argparse.ArgumentParser(description='Get information of the context to visualize')
parser.add_argument('path_to_model',
                    help='Path to the folder where the contexts lie')
args = parser.parse_args()

path_model = Path(os.path.abspath(args.path_to_model))
code_path = Path(__file__).parent.absolute()
motordb_path = code_path.parent.parent / "MotorDB"

train_path_par = motordb_path / "ODETraining.xlsx"
test_path_par = motordb_path / "ODETest.xlsx"

train_path_el = motordb_path / "Training.xlsx"
test_path_el = motordb_path / "Test.xlsx"

train_path_con = path_model / "TrainContexts.xlsx"
test_path_con = path_model / "TestContexts.xlsx"

df_par = pd.read_excel(train_path_par).dropna(how='all', axis='columns')
df_el = pd.read_excel(train_path_el).dropna(how='all', axis='columns')

df_par_te = pd.read_excel(test_path_par).dropna(how='all', axis='columns')
df_el_te = pd.read_excel(test_path_el).dropna(how='all', axis='columns')

df_con = pd.read_excel(train_path_con)
df_con_te = pd.read_excel(test_path_con)

df_stack = pd.concat([df_con, df_par], axis=1)
df_stack_te = pd.concat([df_con_te, df_par_te], axis=1)
df_stack = pd.concat([df_stack, df_stack_te], axis=0)
df_stack_el = pd.concat([df_el, df_el_te], axis=0)
gen_plot( df_stack, df_stack_el.reindex(columns=['Ld', 'Lq', 'Rs', 'p', 'UDC','Psip',  'In', 'Omegan']), "CorrelationMatrix", path_model)
