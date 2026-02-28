import numpy as np
import scipy as sp
import scipy.io as sio

# mPCA analysis
import sys
sys.path.append('/oak/stanford/groups/henderj/nishalps/code/branch2/nptlbraingaterig/code/analysis/Nishal/Sequences/inferred_input')
sys.path.append('/oak/stanford/groups/henderj/nishalps/code/branch2/nptlbraingaterig/code/analysis/Nishal/Utility/')
import decoders
from decoders import *
import importlib
importlib.reload(decoders)


import inferred_input_utils
importlib.reload(inferred_input_utils)
from inferred_input_utils import *

from sklearn.linear_model import LinearRegression

import shared_utils
from shared_utils import *


def get_data_t12_2023_11_22():
    dat = sio.loadmat('/oak/stanford/groups/henderj/nishalps/multieffector/t5.2023.11.22binnedTX_20ms_LW_RW_LA.mat')

    neural_trials = []
    cues_trials = []
    delays_trials = []

    cues_list = dat['cueList']
    cues_list = [c[0][0] for c in cues_list]

    for itrial in range(dat['goTrialEpochs'].shape[0]):

        trial_start_stop = dat['goTrialEpochs'][itrial, :]
        neural_trials += [dat['binnedTX'][trial_start_stop[0]: trial_start_stop[1], :]]
        cues_trials += [cues_list[dat['trialCues'][itrial, 0] - 1]]
        delays_trials += ['']

    neural_trials = [n[:, :192] for n in neural_trials]
    # print([n.shape for n in neural_trials])

    selected_trials = [i for i in range(len(cues_trials)) if 'DO_NOTHING' not in cues_trials[i]]
    neural_trials = [neural_trials[i] for i in selected_trials]
    cues_trials = [cues_trials[i] for i in selected_trials]
    delays_trials = [delays_trials[i] for i in selected_trials]
    session_trials = np.ones((len(neural_trials), 1))

    L = np.min([n.shape[0] for n in neural_trials])
    # L = np.int(0.6 * L)
    neural_trials = [n[:L, :].astype(np.float32) for n in neural_trials]
    channels = neural_trials[0].shape[-1]

    print('Number of trials: %d' % (len(neural_trials)))
    print('time steps: %d' % L)
    print('number of channels: %d' % channels)

    return neural_trials, cues_trials, delays_trials, session_trials

import utils_cvc
mpca_multieffector = utils_cvc.mpca_cvc

def parse_cue(cue):
    arrow_list = [x[0] for x in cue.split(' ') if x.split('\\n')[0] in ['↑', '↓']]
    arrow_dict = {'↑': 1, '↓': -1}

    if len(arrow_list) > 0:
        k =np.array([arrow_dict[x] for x in arrow_list])
    else: 
        k = np.array([0, 0, 0])
    return k

def get_colors_multieffector(cues):

    # kinematic encoding
    kinematic_cue = []
    for c in cues:
        kinematic_cue += [parse_cue(c)]
    
    c0 = np.array([1, 0, 0])
    c1 = np.array([0, 1, 0])
    c2 = np.array([0, 0, 1])

    cols, cols_first, cols_second, cols_third = [], [], [], []
    for k in kinematic_cue:
        
        k_color = (np.array(k) + 1 ) / 2
        cue_col = c0 * k_color[0] + c1 * k_color[1] + c2 * k_color[2]
        col_first = c0 * k_color[0]
        col_second = c1 * k_color[1]
        col_third = c2 * k_color[2]

        cols += [cue_col]
        cols_first += [col_first]
        cols_second += [col_second]
        cols_third += [col_third]

    return np.array(kinematic_cue), cols, cols_first, cols_second, cols_third




def analysis_single_pts_multieffector(feature, cues, markersize=5, method='tdr'):
    
    # get kinematics
    kinematics, cols, cols_first, cols_second, cols_third = get_colors_multieffector(cues)
    

    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        feature_3d = pca.fit_transform(feature)

    else:
        reg = LinearRegression()
        reg.fit(kinematics, feature)
        q, _ = np.linalg.qr(reg.coef_)
        feature_3d = feature.dot(q)

    feature_3d = np.array(feature_3d)

    fig=  plt.figure(figsize=(20, 5))
    for ic, (c, t) in enumerate(zip([cols, cols_first, cols_second, cols_third], 
                                    ['all', 'first', 'second', 'third'])):
        ax = fig.add_subplot(1, 4, ic+1, projection='3d') 
        # ax.view_init(elev=0, azim=75)
        for ipt in range(feature_3d.shape[0]):
            ax.plot3D(np.expand_dims(feature_3d[ipt, 0], 0), 
                    np.expand_dims(feature_3d[ipt, 1], 0), 
                    np.expand_dims(feature_3d[ipt, 2], 0), '.', color=c[ipt])
            ax.set_title(t)
                   