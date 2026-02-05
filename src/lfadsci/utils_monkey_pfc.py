import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import lfadsci.shared_utils
from lfadsci.shared_utils import *

# Working memory data from Moore group
def get_data_monkey_pfc(filename='/oak/stanford/groups/henderj/nishalps/Moore/data/hb_20221024_cells.mat', 
                        T_start=90, 
                        bin_size=10, 
                        remove_duplicate_channels=True):
    '''Get data.
    
    # T_start = 50 if you want to include stim presentation, 80 if we want to start AFTER stim presentation

    '''

    
    data = sio.loadmat(filename)
    neural_trials = []
    cues_trials = []
    delays_trials = []

    frMat = data['frMat']
    frMat_lr = np.zeros((frMat.shape[0], 
                         np.floor(frMat.shape[1] / bin_size).astype(np.int), 
                         frMat.shape[2]))
    for iitime, itime in enumerate(np.arange(0, frMat.shape[1], bin_size)):
        frMat_lr[:, iitime, :] = frMat[:, itime: itime + bin_size, :].sum(-2)
    frMat_lr = frMat_lr[:, T_start:, :]

    nTrials = frMat_lr.shape[0]
    for itrial in range(nTrials):
        neural_trials += [frMat_lr[itrial, :, :]]
        cues_trials += [data['angs'][itrial]]         
        delays_trials += ['']

    # make same length
    L = np.min(np.array([n.shape[0] for n in neural_trials]))
    L = int(L)
    neural_trials = [n[:L, :].astype(np.float32) for n in neural_trials]
    session_trials = np.ones((len(neural_trials), 1))

    if remove_duplicate_channels:
        print('remove duplicate channels')
        frMat_lr, neural_trials = remove_duplicate_channels_fn(frMat_lr, neural_trials)

    return neural_trials, cues_trials, delays_trials, session_trials

def remove_duplicate_channels_fn(frMat_lr, neural_trials):
    # Are there any duplicate channels?
    ff = np.reshape(frMat_lr, [-1, frMat_lr.shape[-1]])
    corr = np.corrcoef(ff.T)

    val = []
    for i in range(corr.shape[0]):
        val += [np.max(corr[i, list(range(i)) +  list(range(i+1, corr.shape[0]))])]
    val = np.array(val)

    mean_act = np.mean(ff, 0)

    corr_cutoff = 0.9
    mean_act_cutoff = 0.05

    '''
    plt.figure()
    plt.plot(mean_act, val, '.')
    for i in range(corr.shape[0]):
        plt.text(mean_act[i], val[i], i)

 
    plt.axhline(corr_cutoff, color='r')
    plt.axvline(mean_act_cutoff, color='r')
    plt.xlabel('mean activity')
    plt.ylabel('correlation')
    plt.title('remove channels that are correlated '
              '(> %.2f) and have mean activity (> %.2f)' % (corr_cutoff, mean_act_cutoff))
    '''
    rejected_channels = np.where(np.logical_and(val > corr_cutoff, mean_act > mean_act_cutoff))[0]
    print('rejected_channels', rejected_channels)
    
    # rejected_channels2 = np.where(mean_act < 0.05)[0]
    # rejected_channels = np.concatenate([rejected_channels, rejected_channels2], axis=0)
    # print('rejected_channels', rejected_channels)

    selected_channels = np.setdiff1d(np.arange(frMat_lr.shape[-1]), rejected_channels)

    frMat_lr = frMat_lr[:, :, selected_channels]
    neural_trials = [n[:, selected_channels] for n in neural_trials]
    
    return frMat_lr, neural_trials


def mpca_monkey_pfc(cues, features, projections=None, dt=10, smoothen_sig=5):
    """
    Perform mPCA (multi-dimensional Principal Component Analysis) on monkey PFC data.

    Parameters:
    cues : list of array-like
        The cues corresponding to each trial.
    features : array-like, shape (n_trials, n_timepoints, n_features)
        The neural features for each trial.
    projections : array-like, optional
        The projections to be used in mPCA.
    dt : int, optional, default: 10
        The time step for downsampling.
    smoothen_sig : int, optional, default: 5
        The sigma value for Gaussian smoothing of the signals.

    Returns:
    projections_used : array-like
        The projections used in the mPCA analysis.
    """
    neural_use = []  # List to store neural data for each cue
    cues_use = []  # List to store the cues

    # Collect neural data and cues
    for iicue, icue in enumerate(cues):
        cues_use.append(icue)
        neural_use.append(features[iicue, ...])

    delays_use = ['' for _ in cues_use]  # List to store delays for each cue

    # Plot figure
    plt.figure()

    # Smoothen neural data if smoothen_sig is greater than 0
    if smoothen_sig > 0:
        neural_ = [smoothen(n, smoothen_sig) for n in neural_use]
    else:
        neural_ = neural_use

    # Perform mPCA analysis
    projections_used = analysis_mpca(delays_use,
                                     cues_use,
                                     neural_, dt=dt,
                                     projections=projections)

    return projections_used


def get_selected_angles(frMat_hr, angs, target_angles):

    x, y = [], []
    for iangle, angle in enumerate(target_angles):
        angle_trials = np.where(np.abs(angs - angle) < 0.01)[0]
        x += [frMat_hr[angle_trials, ...]]
        y += [angle + np.zeros(angle_trials.shape[0])]
    
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    # angle1, angle2 = target_angles
    # # Find trial indices for the target angles
    # angle1_trials = np.where(np.abs(angs - angle1) < 0.01)[0]
    # angle2_trials = np.where(np.abs(angs - angle2) < 0.01)[0]
    # trial_ids = np.concatenate([angle1_trials, angle2_trials], axis=0)

    # # Extract data for the target angles
    # frMat_hr1 = frMat_hr[angle1_trials, ...]
    # frMat_hr2 = frMat_hr[angle2_trials, ...]
    
    # # Combine data and create labels
    # X = np.concatenate([frMat_hr1, frMat_hr2], axis=0)
    # y = np.concatenate([np.zeros(frMat_hr1.shape[0]), np.ones(frMat_hr2.shape[0])], axis=0).astype(np.int)
    return x, y
    
def lapse_analysis(X_train, y_train, X_test, y_test, step_size=1, delta=100, color='r', figures_on=True):
    """
    Perform lapse analysis on the given data using logistic regression.

    Parameters:
    X : array-like, shape (n_samples, n_features, n_channels)
        The input data.
    y : array-like, shape (n_samples,)
        The target labels.
    itest : int
        The index of the test sample.
    step_size : int, optional, default: 1
        The step size for the sliding window.
    delta : int, optional, default: 100
        The width of the sliding window.
    color : str, optional, default: 'r'
        The color of the plot.
    figures_on : bool, optional, default: True
        Whether to display the figures.

    Returns:
    steps : array-like
        The steps of the sliding window.
    prob_log : array-like
        The probabilities of the test sample being correctly classified at each step.
    """

    # Extract the test sample and remove it from the training data
    # X_test, y_test = np.expand_dims(X[itest, ...], 0), y[itest]
    # training_trials = np.setdiff1d(np.arange(X.shape[0]), [itest])
    # X_train, y_train = X[training_trials, ...], y[training_trials]
    
    prob_log = []  # List to store probabilities
    steps = np.arange(0, X_train.shape[1], step_size)  # Define steps for the sliding window

    # Iterate over the steps
    for start_time in steps:
        
        end_time = np.minimum(start_time + delta, X_train.shape[1])  # Calculate end time of the window
        # Train logistic regression on the current window
        clf = LogisticRegression(random_state=0, max_iter=10000).fit(X_train[:, start_time: end_time, :].sum(1), y_train)

        # Predict the log probabilities for the test sample
        y_test_log_proba = clf.predict_log_proba(X_test[:, start_time: end_time, :].sum(1))
        
        # Convert log probability to probability
        y_test_log_proba = np.array([y_test_log_proba[i, y_test[i]] for i in range(y_test_log_proba.shape[0])])
        p_class = np.exp(y_test_log_proba)
        prob_log.append(p_class)  # Append probability to the log

    prob_log = np.array(prob_log).T  # samples x time

    print('prob_log', prob_log.shape)
    # Plot the probabilities if figures_on is True
    if figures_on:
        for isample in range(prob_log.shape[0]):
            
            plt.plot(steps, prob_log[isample, :], color=color)
        plt.xlabel('bin mid time')
        plt.ylabel('test accuracy')
        plt.ylim([0, 1])
        plt.axhline(0.5, ls='--', color='k')

    return steps, np.array(prob_log)


def run_lapse_analysis(frMat_hr_train, angs_train, 
                       frMat_hr_test, angs_test, 
                       target_angles, figures_on=False, step_size=2, delta=10):
    """
    Perform lapse analysis for specified target angles.

    Parameters:
    frMat_hr : array-like, shape (n_trials, n_timepoints, n_features)
        High-resolution firing rate matrix.
    angs : array-like, shape (n_trials,)
        Array of angles corresponding to each trial.
    target_angles : tuple of floats
        The two target angles to analyze.
    figures_on : bool, optional, default: False
        Whether to display figures.
    step_size : int, optional, default: 2
        The step size for the sliding window.
    delta : int, optional, default: 10
        The width of the sliding window.

    Returns:
    prob_log : array-like, shape (n_trials, n_steps)
        The probabilities of the test sample being correctly classified at each step for each trial.
    """
    
    
    if isinstance(angs_train, list):
        angs_train = np.squeeze(np.array(angs_train))
    if isinstance(angs_test, list):
        angs_test = np.squeeze(np.array(angs_test))
    
    def label(x, target_angles):
        return np.where(np.array(target_angles) == x)[0]

    X_train, y_train = get_selected_angles(frMat_hr_train, angs_train, target_angles)
    y_train = np.array([label(y, target_angles) for y in y_train]).squeeze()

    X_test, y_test = get_selected_angles(frMat_hr_test, angs_test, target_angles)
    y_test = np.array([label(y, target_angles) for y in y_test]).squeeze()

    steps, prob_log = lapse_analysis(X_train, y_train, X_test, y_test, step_size=step_size, delta=delta, figures_on=figures_on)

    '''
    prob_log = []  # List to store probabilities for each trial

    # Iterate over all trials to perform lapse analysis
    for itest in tqdm(range(X.shape[0])): 
        if figures_on:
            plt.figure()
            plt.title(f'Trial ID: {trial_ids[itest]}')
        
        # Perform lapse analysis for the current test trial
        steps, prob = lapse_analysis(X, y, itest, step_size=step_size, delta=delta, figures_on=figures_on)
        prob_log.append(prob)
    
    prob_log = np.array(prob_log)  # Convert list of probabilities to an array
    '''
    return steps, prob_log


import colorsys
def convert_cues_to_colors_monkey_pfc(cues):
    # convert cues to colors
    threshold_fn = lambda x: x if x > 0 else x + 360
    color_list = []
    for ielem in range(len(cues)):
        angle = np.rad2deg(cues[ielem])
        angle = threshold_fn(angle)
        color = colorsys.hsv_to_rgb(angle/360, 1.0, 1.0)
        color_list += [color]
        
    return color_list

def analysis_single_pts_pfc(feature, cues, ax=None, method='tdr', plot_means=True, idx_plot=None):
    
    if ax is None:
        ax = plt.gca()
        
    # get colors
    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        feature_2d = pca.fit_transform(feature)
        q = pca.components_

    else:
        from sklearn.linear_model import LinearRegression
        cues_2d = np.array([np.cos(cues), np.sin(cues)]).T
        reg = LinearRegression()
        reg.fit(cues_2d, feature)
        q, _ = np.linalg.qr(reg.coef_)
        feature_2d = feature.dot(q)

#     plt.figure(figsize=(figsize, figsize))
    if plot_means:
        means = []
        for icue in np.unique(cues):
            idx= np.where(cues == icue)[0]
            means += [feature_2d[idx, :].mean(0)]

        means = np.array(means)
        ax.scatter(means[:, 0], means[:,1], 
                    c=np.unique(cues), 
                    s=400, marker='*', cmap='hsv', vmin=-np.pi, vmax=np.pi)

    
    # ax.scatter(feature_2d[:, 0], feature_2d[:, 1], 
    #             c=convert_cues_to_colors_monkey_pfc(cues), 
    #             s=10, alpha=0.5, cmap='hsv')

    
    if idx_plot is None:
        background = False
        idx_plot = np.arange(feature_2d.shape[0]).astype(np.int)
        print(idx_plot)
    else:
        background = True
        idx_plot = np.array(idx_plot).astype(np.int)

    if background: 
        ax.plot(feature_2d[:, 0], 
               feature_2d[:, 1], '.', markersize=2, color=np.ones(3)*0.7)

    ax.scatter(feature_2d[idx_plot, 0], 
               feature_2d[idx_plot, 1], s=10,
                c=cues[idx_plot], 
                cmap='hsv', vmin=-np.pi, vmax=np.pi, alpha=1)

#     plt.colorbar()
    ax.set_xlabel(f'{method} 0')
    ax.set_ylabel(f'{method} 1')
    ax.set_xticks([])
    ax.set_yticks([])
    
    return q

        
def analysis_single_pts_generic(feature, labels, cmap='gray', s=20):
    
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    feature_2d = pca.fit_transform(feature)
           
#     plt.figure(figsize=(figsize, figsize))
    plt.scatter(feature_2d[:, 0], feature_2d[:, 1], 
                c=labels, 
                s=s, cmap=cmap)
    plt.xlabel('pc 0')
    plt.ylabel('pc 1')
    plt.xticks([])
    plt.yticks([])
    
    


def get_updated_states(model, bias, states):
    bias_padded = np.concatenate([bias, 
                      np.zeros((bias.shape[0], model.tv_input_dim))], axis=1)

    updated_states = []
    for itime in tqdm(np.arange(states.shape[1])):
        updated_state = model.generators[0](np.expand_dims(bias_padded, 1), 
                                            initial_state = tf.constant(states[:, itime, :].astype(np.float32)))
        updated_state = updated_state[:, 0, :]
        updated_states += [updated_state]

    updated_states = np.transpose(np.array(updated_states), [1, 0, 2])
    return updated_states
    