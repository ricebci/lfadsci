import scipy.io as sio
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage

import scipy as sp
import scipy.stats

import pickle
import colorsys
import lfadsci.shared_utils
from lfadsci.shared_utils import *



def simulate_pendulam(m, g, l, b, u0, theta_init, time_steps, step_size=0.01):
    
    theta_log = []
    state_log = []
    
    state = [theta_init, 0]  # [\theta, \dot{\theta}]
    for _ in range(time_steps):
        
        theta_log += [state[0]]
        state_log += [np.array(state)]
            
        theta_dotdot = (-b / (m * l**2)) * state[1] - (g / l) * np.sin(state[0]) + u0 / (m * l**2)

        state[0] = state[0] + step_size * state[1]
        state[1] = state[1] + step_size * theta_dotdot
    
    return np.array(theta_log), np.array(state_log)

def sample_neural_activity(latent_act, K, bias, rng):
    
#     rates = latent_act.dot(K) + bias
#     rates = np.maximum(latent_act.dot(K) + bias, 0)
    rates = np.exp(latent_act.dot(K) + bias)
    neural = rng.poisson(rates)
#     print(np.max(rates))
    return neural

def sample_neural_rates(latent_act, K, bias):
    
    rates = np.maximum(latent_act.dot(K) + bias, 0)
#     rates = np.maximum(latent_act.dot(K) + bias, 0)
#     rates = np.exp(latent_act.dot(K) + bias)
#     neural = np.random.poisson(rates)
#     print(np.max(rates))
    return rates



def get_data(n_trials_total=1000, theta_init_randomize=False, b_randomize=False, g_randomize=True):
    # Sample data - one trial per combination
    m = 1
    l = 5
    time_steps = 300

    # Neural activity params
    bin_sz = 20
    neurons = 192

    firing_rate = 125
    scale_k = 1500 /  (1000 / bin_sz) #200 /  (1000 / bin_sz)  
    scale_bias = firing_rate /  (1000 / bin_sz)

    # firing_rate = 25
    # scale_k = 400 /  (1000 / bin_sz) #200 /  (1000 / bin_sz)  
    # scale_bias = firing_rate /  (1000 / bin_sz)


    # NOTE: scale_k determines the amount of firing rate modulation above mean firing rate. 
    # I spent a few weeks increasing the firing rate with bias before realizing that 
    # scale_k was low and hence I was not getting enough modulation.


    # TODO: make sure signal is preserved after this
    rng = np.random.Generator(np.random.PCG64(seed=54))
    K = rng.standard_normal((2, neurons)) * scale_k / np.sqrt(neurons) # scale # 
    bias = np.ones(neurons) * scale_bias  # Needs to be large enough

    neural_rates = []
    cues_trials = []
    delays_trials = []

    g_log = []
    theta_init_log = []
    while True:
        print('.', end='', flush=True)
        
        # TODO: uniform grid instead of random sampling?
        if theta_init_randomize:
            theta_init = rng.choice(np.arange(-2/3, 2/3, 0.01), 1)[0]
        else:
            theta_init = 2/3

        if b_randomize:
            b = rng.choice(np.arange(1, 100, 10), 1)[0]
        else:
            b = 1
        
        if g_randomize:
            g = rng.choice(np.arange(0.5, 4.5, 0.5), 1)[0]
        else:
            g = 1

        u0_by_g = 0
    #     u0_by_g = rng.choice(np.linspace(-30, 30, 1), 1)[0]

        u0 = u0_by_g * g
        theta_init_log += [theta_init]
        g_log += [g]
        theta_log, state_log = simulate_pendulam(m, 
                                                g, 
                                                l, 
                                                b, 
                                                u0, 
                                                theta_init, 
                                                time_steps, 
                                                step_size=0.1)


                    

        rates = sample_neural_rates(state_log, K, bias).astype(np.float32)
        neural_rates += [rates]
        cues_trials += [','.join(['%.1f' % (x) for x in [m, g, l, b, u0, theta_init]])]
        delays_trials += ['']

        '''
        plt.subplot(1, 3, 1)
        plt.plot(state_log[:, 0] / (2 * np.pi))
        plt.title('state[0] (theta)')

        plt.subplot(1, 3, 2)
        plt.plot(state_log[:, 1] / (2 * np.pi))
        plt.title('state[1] (theta dot)')
        '''
            
        if len(neural_rates) > n_trials_total:
            break

    session_trials = np.ones((len(neural_rates), 1))
    neural_trials = rng.poisson(neural_rates).astype(np.float32)   #### REMOVED
    # neural_trials = np.array(neural_rates).astype(np.float32) ####    

    return neural_trials, cues_trials, delays_trials, session_trials



def analysis_single_pts_pendulum(feature, cues, markersize=5, 
                             method='tdr', new_fig=True, ax=None, idx_plot=None):
    
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
        idx_plot_others = np.setdiff1d(np.arange(feature_2d.shape[0]), idx_plot)
        ax.plot(feature_2d[idx_plot_others, 0], 
               feature_2d[idx_plot_others, 1], '.', markersize=1, color=np.ones(3)*0.7)

    ax.scatter(feature_2d[idx_plot, 0], 
               feature_2d[idx_plot, 1], s=markersize,
                c=cues[idx_plot], cmap='hsv', vmin=0, vmax=4.5, alpha=1)




def plot_states_fps(cues, state, fixed_points, n_comps_cis=2, 
                    elev=90, azim=75, ax=None,  alpha_state=0.1, alpha_fp=0.2, 
                    color_fixedpoints=True, fp_marker='.', fp_markersize=10, state_lw=1, 
                   add_cis_dimension=True):
    # get directions to project out.
    partition_use = 'train'

    colors = [colorsys.hsv_to_rgb(c / 5, 0.7, 0.8) for c in cues]
    if color_fixedpoints: 
        colors_fp = [colorsys.hsv_to_rgb(c / 5, 0.7, 0.8) for c in cues]
    else:
        colors_fp = [np.ones(3)*0.5 for c in colors]
    
    mn_state, q_state, feature_selected, _ = low_dim_analysis(cues, 
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


def ev_plot_pendulum(datasets, results, ax=None, sz_scale=0.1, movement_colors=True, cmap='hsv'):
    
    movement_angle = [float(c.split(',')[1]) for c in datasets['train']['cues']]

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
                    cmap=cmap, plot_type='color_size') #, vmin=-np.pi, vmax=np.pi)

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