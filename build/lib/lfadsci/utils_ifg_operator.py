
import matplotlib
import matplotlib.pyplot as plt


import lfadsci.shared_utils
from lfadsci.shared_utils import *

def get_data_verb_conjugation():
    dat = sio.loadmat('/oak/stanford/groups/henderj/nishalps/IFG_operators/verb_conjugation/t12.2023.02.14_binnedTX_20ms.mat')

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
    L = int(0.6 * L)
    neural_trials = [n[:L, :].astype(np.float32) for n in neural_trials]


    print('Number of trials: %d' % (len(neural_trials)))
    print('time steps: %d' % L)
    print('number of channels: %d' %len(channels))

    return neural_trials, cues_trials, delays_trials, session_trials


import scipy
import scipy.io as sio

def get_data_ifg(channels=None, features=['binnedTX', 'spikePower'], task='verb_conjugation'):
    
    if task == 'verb_conjugation':
        dat = sio.loadmat('/oak/stanford/groups/henderj/nishalps/IFG_operators/verb_conjugation/t12.2023.02.14_binnedTX_20ms.mat')
    elif task == 'noun_pluralization':
        dat = sio.loadmat('/oak/stanford/groups/henderj/nishalps/IFG_operators/noun_plularization/t12.2022.12.06_binnedTX_20ms.mat')
    else:
        raise ValueError('Unknown task')
        
    neural_trials = []
    cues_trials = []
    delays_trials = []

    cues_list = dat['cueList']
    cues_list = [str(c)[2:-2] for c in cues_list[0]]
#     print('cues_list', cues_list)
    
    if channels is None: 
        channels = np.arange(dat['binnedTX'].shape[-1])
    

    for itrial in range(dat['delayTrialEpochs'].shape[0]):
    #     print(itrial, dat['delayTrialEpochs'][itrial, :])
        trial_start_stop = dat['delayTrialEpochs'][itrial, :]
        
        neural = []
        if 'binnedTX' in features:
            tx = dat['binnedTX'][trial_start_stop[0]: trial_start_stop[1], channels]
            neural += [tx]
        if 'spikePower' in features:
            lfp = dat['spikePower'][trial_start_stop[0]: trial_start_stop[1], channels]
            neural += [lfp]
            
        neural = np.concatenate(neural, axis=1)

        neural_trials += [neural]
        cues_trials += [cues_list[dat['trialCues'][itrial, 0] - 1]]
        delays_trials += ['']

    # remove do nothing
    selected_trials = [i for i in range(len(cues_trials)) if 'DO_NOTHING' not in cues_trials[i]]
    neural_trials = [neural_trials[i] for i in selected_trials]
    cues_trials = [cues_trials[i] for i in selected_trials]
    delays_trials = [delays_trials[i] for i in selected_trials]


    selected_trials = [i for i in range(len(cues_trials)) if 'end' in cues_trials[i]] #.split('_')[2] == 'start']
    neural_trials = [neural_trials[i] for i in selected_trials]
    cues_trials = [cues_trials[i] for i in selected_trials]
    delays_trials = [delays_trials[i] for i in selected_trials]
    session_trials = np.ones((len(neural_trials), 1))

    L = np.min([n.shape[0] for n in neural_trials])
    neural_trials = [n[:L, :].astype(np.float32) for n in neural_trials]


    print('Number of trials: %d' % (len(neural_trials)))
    print('time steps: %d' % L)
    print('number of channels: %d' % neural_trials[0].shape[-1])

    # return neural_trials, cues_trials, delays_trials, session_trials
    return neural_trials, cues_trials, delays_trials, session_trials


def snip_trials(neural_trials, dt=0.02, t_min=0, t_max=0):
    t_min_idx = np.floor(t_min / dt).astype(np.int64)
    t_max_idx = np.ceil(t_max / dt).astype(np.int64)
    n = [n[t_min_idx: t_max_idx, ...] for n in neural_trials]
    return n

######### mPCA analysis ##########
def mpca_ifg_operator(cues, features, projections=None, dt=20, split_char='_', smoothen_sig=5):
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
    if smoothen_sig > 0:
        neural_ = [smoothen(n, smoothen_sig) for n in neural_use]
    else:
        neural_ = neural_use
    projections_used = analysis_mpca(delays_use,
                                     cues_use,
                                     neural_, dt=dt,
                                     projections=projections)

    return projections_used

def convert_cues_to_idx(cues, cue_idx_map=None):
    if cue_idx_map is None:
        cues_unique = list(set(cues))
        cue_idx_map = dict(zip(cues_unique, range(len(cues_unique))))
    return [cue_idx_map[c] for c in cues]

def analysis_single_pts_ifg(feature, cues, markersize=5, 
                             method='tdr', new_fig=True, ax=None, idx_plot=None, cue_idx_map=None):
    
    cue_idx = np.array(convert_cues_to_idx(cues, cue_idx_map=cue_idx_map)).astype(np.int64)

    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        feature_2d = pca.fit_transform(feature)

    else:
        reg = LinearRegression()
        reg.fit(kinematics, feature)
        q, _ = np.linalg.qr(reg.coef_)
        feature_2d = feature.dot(q)

    if new_fig:
        plt.figure(figsize=(5, 5))
    if ax is None:
        ax = plt.gca()

    if idx_plot is None:
        background = False
        idx_plot = np.arange(feature_2d.shape[0]).astype(np.int64)
        print(idx_plot)
    else:
        background = True
        idx_plot = np.array(idx_plot).astype(np.int64)

    if background: 
        ax.plot(feature_2d[:, 0], 
               feature_2d[:, 1], '.', markersize=1, color=np.ones(3)*0.7)

    print(cue_idx[idx_plot])
    ax.scatter(feature_2d[idx_plot, 0], 
               feature_2d[idx_plot, 1], s=markersize,
                c=cue_idx[idx_plot], #cmap='hsv', vmin=-np.pi, vmax=np.pi, 
                alpha=1)


def ev_plot_ifg(datasets, results, ax=None, sz_scale=0.1, movement_colors=True, cmap='hsv'):
    
    cues = datasets['train']['cues']
    cue_idx = np.array(convert_cues_to_idx(cues, cue_idx_map=None)).astype(np.int64)

    sizes = []
    colors = []
    for ijev, jev in enumerate(results['train']['jacobians']):
        sz = []
        cols = []
        for iijev in range(len(jev)):
            if 'mode_mses' not in results['train'].keys():
                sz +=[sz_scale * np.ones(jev[iijev]['J'].shape[0])]
            else:
                sz += [sz_scale * results['train']['mode_mses'][ijev][iijev]['mse_log_mode_removed']]
                # sz +=[sz_scale * np.ones(jev[iijev]['J'].shape[0])]
#                 sz += [sz_scale * results['train']['mode_mses'][ijev][iijev]['kl_log_mode_removed']]
                
            if movement_colors:
                cols += [cue_idx[ijev] * np.ones(jev[iijev]['J'].shape[0])]
            else:
                cols += [0 * np.ones(jev[iijev]['J'].shape[0])]
                
        sizes += [sz]
        colors += [cols]


    idx_selected = [i for i in range(len(colors)) if len(colors[i]) > 0]
    jacobians = [results['train']['jacobians'][i] for i in idx_selected]
    cols = [colors[i] for i in idx_selected]
    sz = [sizes[i] for i in idx_selected]
    
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
    plot_eigen_vals(jacobians, cols, ax, sz=sz, alpha=1,
                    percentile_lower=0, percentile_higher=100,  dt=20/1000, 
                    cmap=cmap, plot_type='color_size', vmin=0, vmax=22)

#     ax.set_xlim([0.7, 1.1])
# #     # plt.ylim(np.array([-0.3, 0.3]) / (2 * np.pi * 0.005))
#     ax.set_ylim([-5, 5])

    # ax.set_xlim([0.2, 1.1])
    # ax.set_ylim([-5, 5])

    # plt.axhline(1.5)
    # plt.axhline(2.5)
    # plt.axhline(4.0)

    # ax.axhline(1)
    # ax.axhline(0.4)
    # ax.axhline(0.2)