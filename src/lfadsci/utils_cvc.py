import numpy as np
import scipy as sp
import scipy.io as sio

import matplotlib
import matplotlib.pyplot as plt

# mPCA analysis
# import sys
# sys.path.append('/oak/stanford/groups/henderj/nishalps/code/branch2/nptlbraingaterig/code/analysis/Nishal/Sequences/inferred_input')
# sys.path.append('/oak/stanford/groups/henderj/nishalps/code/branch2/nptlbraingaterig/code/analysis/Nishal/Utility/')
# import decoders
# from decoders import *
# import importlib
# importlib.reload(decoders)


# import inferred_input_utils
# importlib.reload(inferred_input_utils)
# from inferred_input_utils import *

from sklearn.linear_model import LinearRegression

import lfadsci.shared_utils
from lfadsci.shared_utils import *

def get_data_t12_2023_06_29():
    dat = sio.loadmat('/oak/stanford/groups/henderj/nishalps/sequences/t12/t12.2023.06.29_binnedTX_20ms.mat')

    neural_trials = []
    cues_trials = []
    delays_trials = []

    cues_list = dat['cueList']
    cues_list = [c[0][0] for c in cues_list]

    for itrial in range(dat['goTrialEpochs'].shape[0]):
        if dat['goodTrialsIdx'][0, itrial] != 1:
            continue

        trial_start_stop = dat['goTrialEpochs'][itrial, :]
        neural_trials += [dat['binnedTX'][trial_start_stop[0]: trial_start_stop[1], :]]
        cues_trials += [cues_list[dat['trialCues'][itrial, 0] - 1]]
        delays_trials += ['']

    # channels = np.append(np.arange(32), np.arange(96, 128))
    # channels = np.append(np.arange(32, 64), np.arange(64, 96))
    channels = np.arange(128)
    neural_trials = [n[:, channels] for n in neural_trials]
    # print([n.shape for n in neural_trials])

    selected_trials = [i for i in range(len(cues_trials)) if 'DO_NOTHING' not in cues_trials[i]]
    neural_trials = [neural_trials[i] for i in selected_trials]
    cues_trials = [cues_trials[i] for i in selected_trials]
    delays_trials = [delays_trials[i] for i in selected_trials]
    session_trials = np.ones((len(neural_trials), 1))

    L = np.min([n.shape[0] for n in neural_trials])
    L = np.int(0.6 * L)
    neural_trials = [n[:L, :].astype(np.float32) for n in neural_trials]


    print('Number of trials: %d' % (len(neural_trials)))
    print('time steps: %d' % L)
    print('number of channels: %d' %len(channels))

    return neural_trials, cues_trials, delays_trials, session_trials


def get_data_t15_2023_10_27(trial_stop_ = 200, channels=np.arange(256)):
    
    dat = sio.loadmat('/oak/stanford/groups/henderj/nishalps/sequences/t15/t15.20231027/binnedTX_spikePow_20ms.mat')

    neural_trials = []
    cues_trials = []
    delays_trials = []

    cues_list = dat['cueList']
    cues_list = [str(c)[2:-2] for c in cues_list[0]]

    for itrial in range(dat['goCue'].shape[1]):

        trial_start = dat['goCue'][0, itrial]
        trial_stop = trial_start + trial_stop_
        neural_trials += [dat['tx'][trial_start: trial_stop, :]]
        cues_trials += [cues_list[dat['trialCue'][0, itrial] - 1]]
        delays_trials += ['']

    neural_trials = [n[:, channels] for n in neural_trials]

    selected_trials = [i for i in range(len(cues_trials)) if 'DO_NOTHING' not in cues_trials[i]]
    neural_trials = [neural_trials[i] for i in selected_trials]
    cues_trials = [cues_trials[i] for i in selected_trials]
    delays_trials = [delays_trials[i] for i in selected_trials]
    session_trials = np.ones((len(neural_trials), 1))

    L = np.min([n.shape[0] for n in neural_trials])

    neural_trials = [n[:L, :].astype(np.float32) for n in neural_trials]


    print('Number of trials: %d' % (len(neural_trials)))
    print('time steps: %d' % L)
    print('number of channels: %d' %len(channels))

    return neural_trials, cues_trials, delays_trials, session_trials

def get_data_t15_2024_03_01(trial_stop_ = 200, channels=np.arange(256)):
    
    dat = sio.loadmat('/oak/stanford/groups/henderj/nishalps/sequences/t15/t15.20240301/binnedTX_spikePow_20ms.mat')

    neural_trials = []
    cues_trials = []
    delays_trials = []

    cues_list = dat['cueList']
    cues_list = [str(c)[2:-2] for c in cues_list[0]]

    for itrial in range(dat['goCue'].shape[1]):

        trial_start = dat['goCue'][0, itrial]
        trial_stop = trial_start + trial_stop_
        neural_trials += [dat['tx'][trial_start: trial_stop, :]]
        cues_trials += [cues_list[dat['trialCue'][0, itrial] - 1]]
        delays_trials += ['']

    neural_trials = [n[:, channels] for n in neural_trials]

    selected_trials = [i for i in range(len(cues_trials)) if 'DO_NOTHING' not in cues_trials[i]]
    neural_trials = [neural_trials[i] for i in selected_trials]
    cues_trials = [cues_trials[i] for i in selected_trials]
    delays_trials = [delays_trials[i] for i in selected_trials]
    session_trials = np.ones((len(neural_trials), 1))

    cues_trials = [c.strip() for c in cues_trials]

    L = np.min([n.shape[0] for n in neural_trials])

    neural_trials = [n[:L, :].astype(np.float32) for n in neural_trials]


    print('Number of trials: %d' % (len(neural_trials)))
    print('time steps: %d' % L)
    print('number of channels: %d' %len(channels))

    return neural_trials, cues_trials, delays_trials, session_trials

######### mPCA analysis ##########
def mpca_cvc(cues, features, projections=None, dt=20, split_char='ah'):
    neural_use = []
    cues_use = []
    for iicue, icue in enumerate(cues):
        if split_char not in icue:
            continue
        jcue = icue.split(split_char)
        cues_use += [jcue]
        neural_use += [features[iicue, ...]]

    delays_use = ['' for icue in cues_use]
    plt.figure()
    projections_used = analysis_mpca(delays_use,
                                     cues_use,
                                     neural_use, dt=dt,
                                     projections=projections)
    # plt.subplot(1, 3, 1)
    # plt.title('CIS')
    #
    # plt.subplot(1, 3, 2)
    # plt.title('first consonant')
    #
    # plt.subplot(1, 3, 3)
    # plt.title('second consonant')

    return projections_used


# get colors and kinematic encoding


def get_colors_cvc_t12(cues):

    consonant_colors = {'K': 0, 'N': 1, 'SH': 0.5}
    kinematic_encoding = {'K': -1, 'N': 1, 'SH': 0}
    split_str = 'ah'

    c0 = np.array([1, 0, 0])
    c1 = np.array([0, 0, 1])
    plot_cues = set(cues)

    cols = []
    cols_first = []
    cols_second = []
    cue_idx = {}
    kinematic_dict = {}
    for iicue, icue in enumerate(plot_cues):
        seq = icue.split(split_str)
        w0 = consonant_colors[seq[0]]
        w1 = consonant_colors[seq[1]]

        #     cue_col = c0 * w0 + c1 * w1
        cue_col = c0 * w0 + c1 * w1
        cols += [cue_col]

        cols_first += [c0 * w0]
        cols_second += [c1 * w1]

        cue_idx.update({icue: iicue})

        kinematic_dict.update({icue: np.array([kinematic_encoding[seq[0]],
                                               kinematic_encoding[seq[1]]])})

    cols = np.array(cols)
    cols_first = np.array(cols_first)
    cols_second = np.array(cols_second)

    return kinematic_dict, cue_idx, cols, cols_first, cols_second

get_colors_cvc = get_colors_cvc_t12

def get_colors_cvc_t15(cues):

    consonant_colors = {'K': 0, 'B': 1}
    kinematic_encoding = {'K': -1, 'B': 1}
    split_str = '-AH-'

    c0 = np.array([1, 0, 0])
    c1 = np.array([0, 0, 1])
    c2 = np.array([0, 1, 0])
    plot_cues = set(cues)

    cols = []
    cols_first = []
    cols_second = []
    cols_third = []
    cue_idx = {}
    kinematic_dict = {}
    for iicue, icue in enumerate(plot_cues):
        seq = icue.split(split_str)
        w0 = consonant_colors[seq[0]]
        w1 = consonant_colors[seq[1]]
        w2 = consonant_colors[seq[2]]

        #     cue_col = c0 * w0 + c1 * w1
        cue_col = c0 * w0 + c1 * w1 + c2 * w2
        cols += [cue_col]

        cols_first += [c0 * w0]
        cols_second += [c1 * w1]
        cols_third += [c2 * w2]

        cue_idx.update({icue: iicue})

        kinematic_dict.update({icue: np.array([kinematic_encoding[seq[0]],
                                               kinematic_encoding[seq[1]], 
                                               kinematic_encoding[seq[2]]])})

    cols = np.array(cols)
    cols_first = np.array(cols_first)
    cols_second = np.array(cols_second)
    cols_third = np.array(cols_third)

    return kinematic_dict, cue_idx, cols, cols_first, cols_second, cols_third


def plot_eigen_vals_cvc(Jev_log, cues, colors):
    plt.figure(figsize=(30, 30))

    cues_conditions = ['NahN', 'SHahN', 'KahN',
                       'NahSH', 'SHahSH', 'KahSH',
                       'NahK', 'SHahK', 'KahK']

    fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    from itertools import chain
    axs = list(chain.from_iterable(axs))

    for icp, cues_plot in enumerate(cues_conditions):  # +
        #                                     [cues_conditions]):

        selected_trials = [itrial for itrial in range(len(cues)) if cues[itrial] in cues_plot]
        Jev_selected = [Jev_log[itrial] for itrial in selected_trials]
        colors_selected = [colors[itrial] for itrial in selected_trials]

        plot_eigen_vals(Jev_selected, colors_selected, axs[icp])
        axs[icp].set_title(cues_plot)
    plt.tight_layout()


def analysis_single_pts_cvc_t12(feature, cues, method='tdr'):

    # get colors
    kinematic_dict, cue_idx, cols, cols_first, cols_second = get_colors_cvc_t12(cues)

    kinematic_cues = np.array([kinematic_dict[icue] for icue in cues])
    cue_idx_log = np.array([cue_idx[iff] for iff in cues])

    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        feature_2d = pca.fit_transform(feature)

    else:
        reg = LinearRegression()
        reg.fit(kinematic_cues, feature)
        q, _ = np.linalg.qr(reg.coef_)
        feature_2d = feature.dot(q)

    plt.figure(figsize=(15, 5))
    iplt = 0
    for col_title, cols_use in zip(['combination', 'first', 'second'], [cols, cols_first, cols_second]):
        iplt += 1
        plt.subplot(1, 3, iplt)
        for icue in np.unique(cue_idx_log):
            plt.plot(feature_2d[cue_idx_log == icue, 0].mean(0),
                     feature_2d[cue_idx_log == icue, 1].mean(0), '*',
                     color=cols_use[icue], markersize=20)

        for ipt in range(feature_2d.shape[0]):
            plt.plot(feature_2d[ipt, 0], feature_2d[ipt, 1], '.',
                     color=cols_use[cue_idx[cues[ipt]]], markersize=10, alpha=0.5)
        plt.title(col_title)


def analysis_single_pts_cvc_t15(feature, cues, markersize=5, method='tdr'):
    
    # get colors
    kinematic_dict, cue_idx, cols, cols_first, cols_second, cols_third = get_colors_cvc_t15(cues)
    kinematic_cues = np.array([kinematic_dict[icue] for icue in cues])

    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        feature_3d = pca.fit_transform(feature)

    else:
        reg = LinearRegression()
        reg.fit(kinematic_cues, feature)
        q, _ = np.linalg.qr(reg.coef_)
        feature_3d = feature.dot(q)

    feature_3d = np.array(feature_3d)

    fig=  plt.figure(figsize=(20, 5))
    for ic, (c, t) in enumerate(zip([cols, cols_first, cols_second, cols_third], 
                                    ['all', 'first', 'second', 'third'])):
        ax = fig.add_subplot(1, 4, ic+1, projection='3d') 
        # ax.view_init(elev=0, azim=75)
        print(feature_3d.shape, c.shape)
        for ipt in range(feature_3d.shape[0]):
            ax.plot3D(np.expand_dims(feature_3d[ipt, 0], 0), 
                    np.expand_dims(feature_3d[ipt, 1], 0), 
                    np.expand_dims(feature_3d[ipt, 2], 0), '.', color=c[cue_idx[cues[ipt]]])
            ax.set_title(t)