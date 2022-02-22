import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn')

SMALL_SIZE = 18
MEDIUM_SIZE = 22
LARGE_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

S2TAG = {'gender': 'gender', 'sexual_orientation':'sex_or', 'race':'race',
         'religion':'rel', 'disability':'disab', 'none':'none'}


def uni_heatmap(index_dict, freq_dict, S, iri2label_dict, ax):
    " Create heatmap"
    # List of top10 from table
    index_list = index_dict[S]
    n_top = len(index_list)
    # Plot frequencies of top10 in all S
    data = pd.DataFrame(freq_dict)
    table = data.loc[data.index.isin(index_list), :]
    table = table.reindex(index_list)
    table.index = [iri2label_dict[iri] for iri in table.index]
    table.columns = [S2TAG[S] for S in table.columns]
    sns.heatmap(table, center=0.5, ax=ax, vmin=0.0, vmax=1.0, xticklabels=True, yticklabels=True)
    ax.set_title('{} {}'.format(S, n_top))
    return n_top


def export_freq_plot(index_dict, freq_dict, iri2label_dict, o_path, title, n_tag,
                      figsize=(6,8)):
    """ Heatmap of frequencies of """
    S = list(index_dict.keys())[0] # prot attribute
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=500)
    plt.tight_layout()

    n_top = uni_heatmap(index_dict, freq_dict, S, iri2label_dict, ax)
    #plt.xticks(rotation=45)

    if o_path is not None:
        filename = '{}_{}top{}_{}.png'.format(title,S,n_top,n_tag)
        fig.savefig(os.path.join(o_path, filename), bbox_inches='tight')


# TODO: multiple plots in one figure (2x3 and sharex)
# seaborn: AttributeError: 'numpy.ndarray' object has no attribute 'spines'
# to try: heatmap with plt https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

# TMP: for some reason not working ax
def export_freq_plots(index_dict, freq_dict, iri2label_dict, o_path, title, n_tag,
                      subplots=None, figsize=(36,6)):
    # y (cols) is second param and can be changed (data.columns)
    S_list = list(index_dict.keys()) # prot attributes
    if not subplots:
        fig, axs = plt.subplots(1, len(S_list), figsize=figsize, sharex=True, dpi=500)
    else:
        fig, axs = plt.subplots(subplots['x'], subplots['y'], figsize=figsize, sharex=True, dpi=500)
    plt.tight_layout()

    # Add to each axis each one of the yvalues
    for i, ax in enumerate(axs):
        S = S_list[i]
        n_top = uni_heatmap(index_dict, freq_dict, S, iri2label_dict, ax)
        # plt.xticks(rotation=45)

    if o_path is not None:
            filename = '{}_{}top{}_{}_multiplot.png'.format(title, S_list, n_top, n_tag)
            fig.savefig(os.path.join(o_path, filename), bbox_inches='tight')