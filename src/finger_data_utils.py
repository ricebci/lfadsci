
# import scipy
import scipy.io as sio
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage

import scipy as sp
import scipy.stats

import pickle
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

def get_data_generator2(neural_trials, session_trials, cues_trials, delays_trials, batch_sz=None, 
                        train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=98):

    
    if train_frac > 1- val_frac - test_frac:
        raise ValueError('train frac bigger than val and test frac')
        
    n_trials = np.array(neural_trials).shape[0]
    train_idx = np.arange(np.int(n_trials * train_frac))
    val_idx = np.arange(np.int(n_trials * (1 - val_frac - test_frac)), 
                       np.int(n_trials * (1 - test_frac)))
    test_idx = np.arange(np.int(n_trials * (1 - test_frac)), 
                         n_trials)
    
    perm = np.random.RandomState(seed=seed).permutation(np.arange(n_trials))
    train_idx = perm[train_idx]
    val_idx = perm[val_idx]
    test_idx = perm[test_idx]
    
    datagenerators = {}
    datasets = {}
    for partition, relevant_idx in zip(['train', 'eval', 'test'], [train_idx, val_idx, test_idx]):
        
        neural_partition = np.array(neural_trials)[relevant_idx, ...]
        session_partition = np.array(session_trials)[relevant_idx, ...]
        cues_partition = [cues_trials[r] for r in relevant_idx]
        delays_partition = [delays_trials[r] for r in relevant_idx]
        
        if batch_sz is None:
            batch_sz = neural_partition.shape[0]
        
        ds_series = tf.data.Dataset.from_tensor_slices((neural_partition, 
                                                        session_partition)).repeat().shuffle(500).batch(batch_sz).repeat()
        datagenerators.update({partition: ds_series})
        datasets.update({partition: {'neural': neural_partition, 'session': session_partition, 
                                     'cues': cues_partition, 'delays': delays_partition, 'trial_ids': relevant_idx}})

    return datagenerators, datasets



def posterior_sample_avg(model, neural, n_samples=50):
    
    # posterior sample and average
    inp_samples = []
    ic_samples = []
    fr_samples = []
    factor_samples = []

    for iiter in range(n_samples):
#         print(iiter)
        op = model.get_factors_ic(neural, 
                                  np.ones((neural.shape[0], 1)), 
                                  get_inputs=True, training=False)
        factors, ic_sample, firing_rate_prediction, factors_stacked, inp_sample = op
        inp_samples += [inp_sample]
        ic_samples += [ic_sample]
        fr_samples += [firing_rate_prediction]
        factor_samples += [factors]


    inp_samples = np.array(inp_samples)
    inp_samples = inp_samples.mean(0)

    ic_samples = np.array(ic_samples)
    ic_samples = ic_samples.mean(0)

    fr_samples = np.array(fr_samples)
    fr_samples = fr_samples.mean(0)
    
    factor_samples = np.array(factor_samples)
    factor_samples = factor_samples.mean(0)
    
    return inp_samples, ic_samples, fr_samples, factor_samples

# no bias input
def posterior_sample_avg_noinp(model, neural, n_samples=50):
    
    # posterior sample and average
    ic_samples = []
    fr_samples = []
    factor_samples = []

    for iiter in range(n_samples):
#         print(iiter)
        op = model.get_factors_ic(neural, 
                                  np.ones((neural.shape[0], 1)), 
                                  training=False)
        factors, ic_sample, firing_rate_prediction, factors_stacked = op
        ic_samples += [ic_sample]
        fr_samples += [firing_rate_prediction]
        factor_samples += [factors]

    ic_samples = np.array(ic_samples)
    ic_samples = ic_samples.mean(0)

    fr_samples = np.array(fr_samples)
    fr_samples = fr_samples.mean(0)
    
    factor_samples = np.array(factor_samples)
    factor_samples = factor_samples.mean(0)
    
    return ic_samples, fr_samples, factor_samples


import scipy
def smoothen(x, sigma=5):
    xx = []
    for idim in range(x.shape[1]):
        xx += [scipy.ndimage.gaussian_filter(x[:, idim], sigma)]
    xx = np.array(xx)
    return xx.T


def get_two_effectors_features(cues, features_plot, smoothen_sig=0):
    labels = {'0.5->0.0': -1, 
              '0.5->0.5': 0, 
              '0.5->1.0': 1}


    feature_selected = []
    kinematics = []
    for finger0 in ['0.5->0.0', '0.5->0.5', '0.5->1.0']:
        for finger1234 in ['0.5->0.0', '0.5->0.5', '0.5->1.0']:
            if (finger0 == '0.5->0.5') and (finger1234 == '0.5->0.5'):
                continue
            for itrial in range(len(cues)):
                if ((cues[itrial][0] == finger0) and 
                    (cues[itrial][1] == finger1234) and 
                    (cues[itrial][2] == finger1234) and
                    (cues[itrial][3] == finger1234) and
                    (cues[itrial][4] == finger1234)):
                    
                    if smoothen_sig == 0:
                        feat = features_plot[itrial, ...]
                    elif smoothen_sig > 0:
                        feat = smoothen(features_plot[itrial, ...], smoothen_sig)
                        
                    feature_selected += [feat]
                    kinematics += [[labels[finger0], labels[finger1234]]]

    feature_selected = np.array(feature_selected)
    kinematics = np.array(kinematics)
    return feature_selected, kinematics
    
def feature_analysis_two_effectors(cues, features_plot, end_tm=-10, n_comps_cis=3, smoothen_sig=0, trial_ids=None,
                                  method='tdr'):    

    import copy 
    
    # get features
    feature_selected, kinematics = get_two_effectors_features(cues, features_plot, smoothen_sig=smoothen_sig)
    
    
    # remove mean 
    mn = np.reshape(feature_selected, [-1, feature_selected.shape[-1]]).mean(0)
    feature_selected_mean_removed = copy.deepcopy(feature_selected) - mn
    
    P = np.eye(mn.shape[0])
    if trial_ids is not None:
        trials_selected, kinematics = get_two_effectors_features(cues, 
                                                                 np.expand_dims(trial_ids, 1))
        
        reg = LinearRegression()
        print(trials_selected.shape, feature_selected_mean_removed.mean(1).shape)
        reg.fit(trials_selected, 
                feature_selected_mean_removed.mean(1))
        print(reg.coef_.shape)
        q, _ = np.linalg.qr(reg.coef_)
        print(q.shape)
        P_trial = np.eye(q.shape[0]) - q.dot(q.T)
        P = P.dot(P_trial)

    # remove CIS
    feature_selected_mean_removed = copy.deepcopy(feature_selected) - mn
    feature_selected_mean_removed = np.einsum('ijk,kl->ijl', feature_selected_mean_removed, P)
    cis = feature_selected_mean_removed.mean(0)
    u, s, v = np.linalg.svd(cis.T)
    P_cis_orth = np.eye(u.shape[0]) - u[:, :n_comps_cis].dot(u[:, :n_comps_cis].T)
    P = P.dot(P_cis_orth)

    # learn TDR dims
    feature_sel_ = copy.deepcopy(feature_selected)
    feature_sel_ = feature_sel_[:, end_tm:, :].mean(-2)
    feature_sel_ -= mn
    feature_sel_ = feature_sel_.dot(P)
        
    print(method)
    if method == 'tdr':
        print('TDR')
        reg = LinearRegression()
        reg.fit(kinematics, feature_sel_)
        q, _ = np.linalg.qr(reg.coef_)
    elif method == 'pca':
        print('PCA')
        pca = PCA(n_components=2)
        pca.fit(feature_sel_)
        q = pca.components_.T
        print(q.shape)

    q = P_cis_orth.dot(q)
    
    features = feature_selected
    return mn, q, features, kinematics

def plot_tdr(feature_selected, kinematics, mn, q, new_fig=True, ls='-', ms='.', alpha=1, cols=None):
    unique_kin = np.unique(kinematics, axis=0)

#     rng = np.random.default_rng(55)
#     cols = rng.uniform(size=(unique_kin.shape[0], 3)) # NOTE 03/31/23: HSV interpolation.
    if cols is None:
        cols = (np.expand_dims(np.array([1, 0, 0]), 1) * (unique_kin[:, 0] + 1 ) /2 + 
            np.expand_dims(np.array([0, 0, 1]), 1) * (unique_kin[:, 1] + 1 ) /2).T

    
    if new_fig:
        plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    for itrial in range(feature_selected.shape[0]):
        feat2d = (feature_selected[itrial, :, :] - mn).dot(q)

        ic = cols[np.argmin(np.sum(np.abs(unique_kin - kinematics[itrial, :]), 1)), :]
        plt.plot(feat2d[:, 0], feat2d[:, 1], color=ic, ls=ls)

    plt.axis('equal')   
    plt.title('single trials')


    plt.subplot(1, 2, 2)
    feat2d_log = []
    for itrial in range(feature_selected.shape[0]):
        feat2d = (feature_selected[itrial, :, :] - mn).dot(q)
        feat2d_log += [feat2d]

    feat2d_log = np.array(feat2d_log)

    for ikin in range(unique_kin.shape[0]):
        ic = np.where(np.sum(np.abs(unique_kin[ikin, :] - kinematics), 1) == 0)[0]
        feat_avg_traj = feat2d_log[ic, ...].mean(0)
        plt.plot(feat_avg_traj[:, 0], feat_avg_traj[:, 1], linewidth=3, color=cols[ikin], ls=ls, alpha=alpha)
        plt.plot(feat_avg_traj[0, 0], feat_avg_traj[0, 1], ms, markersize=30, color=cols[ikin], alpha=alpha)

    plt.axis('equal')
    plt.title('trial average')

