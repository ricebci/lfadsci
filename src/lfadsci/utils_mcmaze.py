
import numpy as np
import sys

sys.path.append('/oak/stanford/groups/henderj/nishalps/neural_latents/nlb_tools')
import nlb_tools
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5

import matplotlib
import matplotlib.pyplot as plt
import importlib

import colorsys

# import inferred_input_utils
# importlib.reload(inferred_input_utils)
# from inferred_input_utils import *

from sklearn.linear_model import LinearRegression

import lfadsci.shared_utils
from lfadsci.shared_utils import *

def get_data(bin_width=5, total_length=900, pre_length=450):

    dataset_name = 'mc_maze'
    datapath = '/oak/stanford/groups/henderj/nishalps/neural_latents/MC_Maze/sub-Jenkins/'
    # datapath = '/oak/stanford/groups/henderj/nishalps/maze_data/sub-Jenkins_ses-20090918_behavior+ecephys.nwb'
    # datapath = '/oak/stanford/groups/henderj/nishalps/maze_data/'
    dataset = NWBDataset(datapath, split_heldout=False)
    dataset.resample(bin_width)  # binning width

    data_trialized = dataset.make_trial_data(start_field='move_onset_time', end_field='end_time', margin=pre_length)

    spikes = data_trialized['spikes'].to_numpy()
    hand_pos = data_trialized['hand_pos'].to_numpy()
    hand_vel = data_trialized['hand_vel'].to_numpy()
    trialid = data_trialized['trial_id'].to_numpy()

    trial_delay = dataset.trial_info['delay'].to_numpy()
    trial_number = dataset.trial_info['trial_id'].to_numpy()
    target_pos = dataset.trial_info['target_pos'].to_numpy()
    barrier_pos = dataset.trial_info['barrier_pos'].to_numpy()


    spikes_trials = []
    hand_pos_trials = []
    hand_vel_trials = []
    label_trials = []
    for itrial in np.unique(trialid):
        if trial_delay[trial_number == itrial] > 400:  # delay longer than 400ms
            spikes_trials += [spikes[trialid == itrial, ...]]
            hand_pos_trials += [hand_pos[trialid == itrial, ...]]
            hand_vel_trials += [hand_vel[trialid == itrial, ...]]
            label_trials += ['%s_%s' % (str(target_pos[trial_number == itrial]), 
                                        str(barrier_pos[trial_number == itrial]))]
        
    if total_length is None:
        L = np.min([s.shape[0] for s in spikes_trials])
    else:
        L = int(total_length / bin_width)
    spikes_trials = [s[:L, ...] for s in spikes_trials]
    hand_pos_trials = [s[:L, ...] for s in hand_pos_trials]
    hand_vel_trials = [s[:L, ...] for s in hand_vel_trials]

    spikes_trials = np.array(spikes_trials).astype(np.float32)
    hand_pos_trials = np.array(hand_pos_trials).astype(np.float32)
    hand_vel_trials = np.array(hand_vel_trials).astype(np.float32)

    print('Number of trials: %d' % spikes_trials.shape[0])
    print('Time steps: %d' % spikes_trials.shape[1])
    print('Number of channels: %d' % spikes_trials.shape[2])

    return spikes_trials, hand_pos_trials, hand_vel_trials, label_trials



def analysis_single_pts_maze(feature, cues, t_start=10, t_end=25, markersize=5, 
                             method='tdr', new_fig=True, ax=None, idx_plot=None):
    
    # get kinematics
    kinematics = np.array([c[t_start: t_end].mean(0) for c in cues])
    movement_angle = np.arctan2(kinematics[:, 0], kinematics[:, 1])

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
        idx_plot = np.arange(feature_2d.shape[0]).astype(np.int)
        print(idx_plot)
    else:
        background = True
        idx_plot = np.array(idx_plot).astype(np.int)

    if background: 
        ax.plot(feature_2d[:, 0], 
               feature_2d[:, 1], '.', markersize=1, color=np.ones(3)*0.7)

    ax.scatter(feature_2d[idx_plot, 0], 
               feature_2d[idx_plot, 1], s=markersize,
                c=movement_angle[idx_plot], cmap='hsv', vmin=-np.pi, vmax=np.pi, alpha=1)


def oscillation_analysis(spikes_trials, label_trials, 
                         times=None, tstart=None, tend=None, dt=None, subtract_cc_mean=False, 
                         smoothen_sig=5, soft_normalize=5, num_pcs=6, num_jpcs=6):
    # look at jPCA

    # do trial averaging
    spikes_trial_averaged = []

    trial_labels = list(set(label_trials))
    for ref_trial in trial_labels:
    #     print(ref_trial)
        trials_selected = [itrial for itrial in range(len(spikes_trials)) if label_trials[itrial] == ref_trial]
        spikes_selected = spikes_trials[trials_selected, ...]
        spikes_selected = np.array([smoothen(spikes_selected[i, :, :], sigma=smoothen_sig) for i in range(spikes_selected.shape[0])])
        spikes_selected = np.nanmean(spikes_selected, 0)
        spikes_trial_averaged += [spikes_selected]


    sys.path.append('/oak/stanford/groups/henderj/nishalps/jPCA')
    import jPCA
    import importlib
    importlib.reload(jPCA)

    jpca = jPCA.JPCA(num_jpcs=num_jpcs)
    from jPCA.util import load_churchland_data, plot_projections

    # Fit the jPCA object to data
    # (projected, 
    #  full_data_var,
    #  pca_var_capt,
    #  jpca_var_capt) = jpca.fit(spikes_trial_averaged, 
    #                            times=range(-180, 450, 5), tstart=-180, tend=450-5)
    (projected, 
     full_data_var,
     pca_var_capt,
     jpca_var_capt) = jpca.fit(spikes_trial_averaged, 
                               times=times, tstart=tstart, tend=tend, 
                               subtract_cc_mean=subtract_cc_mean, 
                               soft_normalize=soft_normalize, num_pcs=num_pcs)

    if dt is not None:
        print(f'tstart {tstart}ms, tend {tend}ms, eigen frequencies (Hz)', np.unique(np.abs(np.imag(jpca.eigvals) / (2 * np.pi * dt))))
    
    print('jpca_var_capt', jpca_var_capt)

    # Plot the projected data
    fig = plt.figure(figsize=(15, 4))
    for iplt, (x, y) in enumerate([(0, 1), (2, 3), (4, 5)]):
        ax = fig.add_subplot(1, 3, iplt+1)
        plot_projections(projected, 
                         x_idx=x,
                         y_idx=y,
                         axis=ax,
                         arrows=True,
                         circles=True,
                         arrow_size=0.001,
                         circle_size=0.001)


    # plt.figure(figsize=(25, 5))
    # for spk in spikes_trial_averaged:
    #     u, s, v = np.linalg.svd(spk)
    #     for isvd in range(5):
    #         plt.subplot(1, 5, isvd + 1)
    #         _ = plt.plot(u[:, isvd], color='r')

    # for isvd in range(5):
    #     plt.subplot(1, 5, isvd + 1)
    #     plt.axvline(color='k', ls='--')
    #     plt.xlabel('time from movement onset')
    #     plt.title(f'SVD {isvd}')

        # plt.axvline(-50, color='g')

# get hues
def get_hues(hand_vel_trials):
    t_start, t_end = 40, 60
    kinematics = np.array([c[t_start: t_end].mean(0) for c in hand_vel_trials])
    movement_angle = np.arctan2(kinematics[:, 0], kinematics[:, 1])
    min_angle, max_angle = np.min(movement_angle), np.max(movement_angle)

    hue = (movement_angle - min_angle)/(max_angle - min_angle)
    return hue

def get_movement_angle(cues, t_start=40, t_end=60):
    kinematics = np.array([c[t_start: t_end].mean(0) for c in cues])
    movement_angle = np.arctan2(kinematics[:, 0], kinematics[:, 1])
    return movement_angle


# lets look at some PSTHs
def plot_psths(spikes_trials, hand_vel_trials, label_trials, axs=None, saturation=1, brightness=1):
    channels = np.argsort(spikes_trials.mean(0).mean(0))[::-1][:3]

    smoothen_sig = 5
    plt.figure(figsize=(15, 5))

    hue = get_hues(hand_vel_trials)
    import colorsys
    
    if axs is None:
        axs = []
        axs += [plt.subplot(1, 3, 1)]
        axs += [plt.subplot(1, 3, 2)]
        axs += [plt.subplot(1, 3, 3)]
    

    for ilabel in list(set(label_trials)):
        trials = [i for i in range(len(label_trials)) if label_trials[i] == ilabel]
        avg_hue = np.mean(hue[trials])

        psths = spikes_trials[np.array(trials), ...].mean(0)
        psths  = smoothen(psths, smoothen_sig)

        for iichann,  ichann in enumerate(channels):
            
            axs[iichann].plot(np.arange(-180, 630 - 180, 5) * 0.005, 
                              psths[:, ichann] / (0.005), lw=1, 
                              color=colorsys.hsv_to_rgb(avg_hue, saturation, brightness))
#             axs[iichann].set_title(f'channnel {ichann}')
            axs[iichann].axvline(0, color='k', ls='--', lw=1)
            axs[iichann].set_ylim([0, 100])
            
            axs[iichann].set_xlabel('time (s)', fontsize=7)
            
            axs[iichann].tick_params(axis='x', labelsize=7)
            
            if iichann == 0:
                axs[iichann].set_ylabel('Hz', fontsize=7)
                axs[iichann].tick_params(axis='y', labelsize=7)
            else:
                axs[iichann].set_yticks([])
        
            
            axs[iichann].grid(False)

            axs[iichann].spines['top'].set_visible(False)
            axs[iichann].spines['right'].set_visible(False)
#             axs[iichann].spines['left'].set_visible(False)
#             axs[iichann].spines['bottom'].set_visible(False)

# plot positions
def plot_positions(hand_pos_trials, hand_vel_trials, ax=None, lw=1, alpha=0.05, 
                   saturation=1, brightness=1):
    import colorsys
    
    if ax is None:
        ax = plt.gca()
    
    hue = get_hues(hand_vel_trials)

    for p in range(hand_pos_trials.shape[0]):
        _ = ax.plot(hand_pos_trials[p, :, 0], 
                     hand_pos_trials[p, :, 1], 
                     color=colorsys.hsv_to_rgb(hue[p], saturation, brightness), 
                     alpha=alpha, lw=lw)
        
    ax.set_xlabel('position x', fontsize=7)
    ax.set_ylabel('position y', fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.text(-200, 100, 'A', fontsize=7)


def plot_states_fps(cues, state, fixed_points, n_comps_cis=2, 
                    elev=90, azim=75, ax=None,  alpha_state=0.1, alpha_fp=0.2, 
                    color_fixedpoints=True, fp_marker='.', fp_markersize=10, state_lw=1, 
                   add_cis_dimension=True):
    # get directions to project out.
    partition_use = 'train'

    t_start, t_end = 40, 60
    kinematics = np.array([c[t_start: t_end].mean(0) for c in cues])
    movement_angle = np.arctan2(kinematics[:, 0], kinematics[:, 1])
    movement_angle = np.array([np.degrees(m) for m in  movement_angle])
    movement_angle = np.where(movement_angle < 0, movement_angle + 360, movement_angle)
    colors = [colorsys.hsv_to_rgb(m/360, 0.7, 0.8) for m in movement_angle]
    if color_fixedpoints: 
        colors_fp = [colorsys.hsv_to_rgb(m/360, 1, 0.8) for m in movement_angle]
    else:
        colors_fp = [np.ones(3)*0.5 for c in colors]
    
    mn_state, q_state, feature_selected, cues = low_dim_analysis(kinematics, 
                                                                 state, 
                                                                 n_comps_cis=n_comps_cis, # project out two CIS direction
                                                                 smoothen_sig=0, 
                                                                 trial_ids = None, #np.expand_dims(datasets[partition_use]['trial_ids'], 1), 
                                                                 do_trial_avg=False, 
                                                                 start_tm=0, end_tm=np.inf, 
                                                                 collapse_time=False, 
                                                                 n_comps_pca=2, method='pca', 
                                                                 add_cis_dimension=add_cis_dimension)

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d') 
    ax.view_init(elev=elev, azim=azim)  # you can play with these parameters to get a better view of the data.


    plot_states_fps_all_trials(state, 
                               fixed_points, 
                               colors, colors_fp, 
                               {'mn': mn_state, 'q': q_state}, 
                               ax, alpha_state=alpha_state, alpha_fp=alpha_fp, 
                               fp_marker=fp_marker, fp_markersize=fp_markersize, state_lw=state_lw)



def ev_plot_maze(datasets, results, ax=None, sz_scale=0.1, movement_colors=True, cmap='hsv'):
    
    t_start, t_end = 40, 60
    kinematics = np.array([c[t_start: t_end].mean(0) for c in datasets['train']['delays']])
    movement_angle = np.arctan2(kinematics[:, 0], kinematics[:, 1])

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
                cols += [movement_angle[ijev] * np.ones(jev[iijev]['J'].shape[0])]
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
                    percentile_lower=0, percentile_higher=100,  dt=5/1000, 
                    cmap=cmap, plot_type='color_size', vmin=-np.pi, vmax=np.pi)

#     ax.set_xlim([0.7, 1.1])
# #     # plt.ylim(np.array([-0.3, 0.3]) / (2 * np.pi * 0.005))
#     ax.set_ylim([-5, 5])

    ax.set_xlim([0.2, 1.1])
    ax.set_ylim([-5, 5])

    # plt.axhline(1.5)
    # plt.axhline(2.5)
    # plt.axhline(4.0)

    # ax.axhline(1)
    # ax.axhline(0.4)
    # ax.axhline(0.2)