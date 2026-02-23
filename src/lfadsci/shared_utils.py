'''Code for fixed point finding and plotting. Shared between datasets.'''
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

onp = np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Model fitting and evaluation utils
def get_data_generator2(neural_trials, session_trials, cues_trials, delays_trials, batch_sz=None,
                        train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=98, include_mask=False):
    if train_frac > 1 - val_frac - test_frac:
        raise ValueError('train frac bigger than val and test frac')

    if include_mask:
        length_max = np.max([n.shape[0] for n in neural_trials])
        masks = np.zeros((len(neural_trials), length_max))
        neural_trials_padded = []
        for iin, n in enumerate(neural_trials):
            nn = np.zeros((length_max, n.shape[-1]))
            nn[:n.shape[0], ...] = n
            neural_trials_padded += [nn.astype(np.float32)]
            masks[iin, :n.shape[0]] = 1
        neural_trials = neural_trials_padded
        masks = masks.astype(np.bool)

    n_trials = np.array(neural_trials).shape[0]
    train_idx = np.arange(int(n_trials * train_frac))
    val_idx = np.arange(int(n_trials * (1 - val_frac - test_frac)),
                        int(n_trials * (1 - test_frac)))
    test_idx = np.arange(int(n_trials * (1 - test_frac)),
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
        if include_mask:
            mask_partition = masks[relevant_idx, ...]
            
        if batch_sz is None:
            batch_sz = neural_partition.shape[0]

        if include_mask:
            ds_series = tf.data.Dataset.from_tensor_slices((neural_partition,
                                                            session_partition, mask_partition)).repeat().shuffle(2000).batch(batch_sz).repeat()
        else:
            ds_series = tf.data.Dataset.from_tensor_slices((neural_partition,
                                                            session_partition)).repeat().shuffle(2000).batch(batch_sz).repeat()

            # trial_len = np.array(neural_trials).shape[1]
            # window_len = int(trial_len / 2)
            # def preprocess(x, y):
                
            #     # start, end = np.sort([np.random.randint(trial_len), np.random.randint(trial_len)])
            #     start = tf.random.uniform((), minval=0, maxval=trial_len-1, dtype=tf.dtypes.int32)
            #     end = tf.random.uniform((), minval=start+1, maxval=trial_len, dtype=tf.dtypes.int32)
            #     print(start, end)
            #     return x[:, start: end, :], y
            
            # ds_series = tf.data.Dataset.from_tensor_slices((neural_partition,
                                                            # session_partition)).repeat().shuffle(2000).batch(batch_sz).map(preprocess).repeat()
            
        datagenerators.update({partition: ds_series})
        datasets.update({partition: {'neural': neural_partition, 'session': session_partition,
                                     'cues': cues_partition, 'delays': delays_partition, 'trial_ids': relevant_idx}})
        if include_mask:
            datasets[partition].update({'mask': mask_partition})

    return datagenerators, datasets


def combine_datasets(data_list, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=98, batch_sz=512, data_weight=None):
    """
    Combine multiple datasets into a single dataset for training, validation, and testing.

    Parameters:
    data_list (list): A list of dictionaries, where each dictionary contains 'neural', 'session_id', and 'cues' keys.

    Returns:
    tuple: A tuple containing:
        - datagenerator_combined (dict): A dictionary with keys 'train', 'test', 'eval', containing combined datasets.
        - datasets_list (list): A list of individual datasets.
    """
    datagenerator_list = []
    datasets_list = []

    # Iterate over each dataset in the provided list
    for idata, data in enumerate(data_list):
        datagenerators, datasets = get_data_generator2(
            data['neural'], 
            data['session_id'].astype(np.int32),  # Convert session_id to int32
            data['cues'],
            data['delays'],  # Placeholder list of empty strings for each neural data entry
            batch_sz=batch_sz,  # Batch size
            train_frac=train_frac,  # Fraction of data for training
            val_frac=val_frac,  # Fraction of data for validation
            test_frac=test_frac,  # Fraction of data for testing
            seed=seed  # Random seed for reproducibility
        )
        datagenerator_list.append(datagenerators)
        datasets_list.append(datasets)

    if data_weight is None:
        data_weight = np.ones(len(datagenerator_list)).astype(np.float32)

    print(f'Original data weights: {data_weight}') 

    # scaling data_weight by the number of samples in the dataset
    data_weight_scaled = []
    for i in range(len(data_list)):
        data_weight_scaled += [data_weight[i] /len(data_list[i]['neural'])]
    data_weight_scaled = np.array(data_weight_scaled)
    data_weight_scaled = data_weight_scaled / np.sum(data_weight_scaled)
    print(f'Rescaled data weights: {data_weight_scaled} to account for  different number of samples accross datasets.')

    # create approximate number of repeats.
    data_weight_int = np.ceil(np.array(data_weight_scaled) * 100).astype(np.int64)
    print(f'Number of repeats per dataset: {data_weight_int}')

    datagenerator_combined = {}
    for partition in ['train', 'test', 'eval']:
        '''
        from IPython import embed; embed()
        combined_dataset = tf.data.Dataset.sample_from_datasets(
                                [data[partition] for data in datagenerator_list], weights=data_weight)

        '''
        # older style.
        # Combine datasets using interleave
        data_list_weighted = []
        for idata, data in enumerate(datagenerator_list):
            print(f'Repeating data {idata}: {data_weight_int[idata]} times')
            data_list_weighted += [data[partition].repeat(data_weight_int[idata])]

        combined_dataset = tf.data.Dataset.from_tensor_slices(data_list_weighted)
        combined_dataset = combined_dataset.interleave(
            lambda x: x,
            cycle_length=len(datagenerator_list),
            block_length=1
        )

        # Shuffle and repeat the combined dataset
        combined_dataset = combined_dataset.shuffle(buffer_size=100).repeat()
        datagenerator_combined.update({partition: combined_dataset})

    return datagenerator_combined, datasets_list


def posterior_sample_avg_new(model, neural, session_id, n_samples=50):

    # posterior sample
    print(session_id)
    op = []
    for iiter in tqdm(range(n_samples)):
        op_iiter = model.run(neural, session_id, training=False, return_numpy=True)

        if iiter == 0:
            output_means = op_iiter
            for ikey in output_means.keys():
                output_means[ikey] = np.array(output_means[ikey])
        else:
            for ikey in op_iiter.keys():
                output_means[ikey] = (output_means[ikey] * iiter  + np.array(op_iiter[ikey]) ) / (iiter + 1)

    # # average
    # output_means = {}
    # for ikey in op[0].keys():
    #     output_means.update({ikey: np.array([o[ikey] for o in op]).mean(0)})

    return output_means


def compile_results(model, datasets, n_samples=100, compute_kinematic_r2=True, partitions=['train', 'eval', 'test']):
    results = {}
    for partition_use in partitions:
        output = posterior_sample_avg_new(model, datasets[partition_use]['neural'], 
                                          np.array(datasets[partition_use]['session'][0, 0]),
                                          n_samples=n_samples)
        output.update({'state': output['states_stacked'][0, ...]})

        # if 'bias_sample' in output.keys():
        #     bias_sample = output['bias_sample']
        # else:
        #     bias_sample = np.zeros((datasets[partition_use]['neural'].shape[0], model.bias_dim))
        # bias_time_sample = np.zeros((1, datasets[partition_use]['neural'].shape[1], 1)) + tf.expand_dims(bias_sample, 1)

        # if 'tv_input_sample' in output.keys():
        #     tv_input_sample = output['tv_input_sample']
        # else:
        #     tv_input_sample = np.zeros((datasets[partition_use]['neural'].shape[0], model.tv_input_dim))
            
        # all_inputs_sample = tf.concat([bias_time_sample, tv_input_sample], axis=2)
        
        # state_traj = model.generators[0](tf.convert_to_tensor(bias_time), 
        #                              initial_state=model.ic_to_state[0](output['ic_sample']), training=False)
        # states_all = np.squeeze(state_traj.numpy())
        # output.update({'state': output})

        results.update({partition_use: output})
        
        # Metrics
        metrics = {}
        # elbo
        elbo, losses = model.get_elbo(datasets[partition_use]['neural'], 
                                 np.array(datasets[partition_use]['session'][0, 0]), 
                                 training=False, n_resamples=n_samples)
        metrics.update({'elbo': elbo, 'losses': losses})
        
        # Prediction cross-correlation
        cc = np.corrcoef(np.ndarray.flatten(datasets[partition_use]['neural']), 
                         np.ndarray.flatten(output['firing_rate']))[0, 1]
        metrics.update({'cc': cc})
        
        # convert quatities to 2D
        convert_to_2d = lambda x: np.reshape(x, [-1, x.shape[-1]])
        neural_2d = convert_to_2d(datasets[partition_use]['neural'])
        mean_fr = neural_2d.mean(0)
        fr_sample_2d = convert_to_2d(output['firing_rate'])
        cues_2d = convert_to_2d(np.array(datasets[partition_use]['delays']))

        # Prediction LL
        ll_model = np.nanmean(fr_sample_2d - neural_2d * np.log(fr_sample_2d))
        ll_constant = np.nanmean(mean_fr - neural_2d * np.log(mean_fr))
        LL = np.mean(ll_model - ll_constant)
        metrics.update({'LL': LL})

        # Classification analysis
        u, y = np.unique(datasets[partition_use]['cues'], return_inverse=True)

        feature_names = ['ic_sample', 'firing_rate', 'factors', 'bias_sample', 'tv_input_sample', 'state']
        
        if partition_use == 'train':
            classifiers = {}

        for feature_name in feature_names:

            if feature_name not in output.keys():
                continue
            
            feature = output[feature_name]
            if len(feature.shape) == 3:
                feature = feature.mean(1)

            # print(feature_name, 'feature', feature.shape, 'y', y.shape)
            if partition_use == 'train':
                classifiers.update({feature_name: LogisticRegression(max_iter=20000)})
                classifiers[feature_name].fit(feature, y)

            y_pred = classifiers[feature_name].predict(feature)
            accuracy = accuracy_score(y, y_pred)
            print(f'{feature_name} Accuracy: {accuracy}')
            metrics.update({f'cue_classification_acc_from_{feature_name}': accuracy})

        # Kinematics R2
        if compute_kinematic_r2:
            if partition_use == 'train':
                import sklearn
                linreg = sklearn.linear_model.LinearRegression()
    #             linreg = sklearn.linear_model.Ridge(alpha=10)
                linreg.fit((fr_sample_2d), cues_2d)
            cues_2d_pred = linreg.predict((fr_sample_2d))
            kinematic_r2 = sklearn.metrics.r2_score(cues_2d, cues_2d_pred)
            metrics.update({'kinematics_r2': kinematic_r2, 'cues_2d_pred': cues_2d_pred, 'cues_2d': cues_2d})
            
            print(f'Partition: {partition_use} Elbo: %.2f,Correlation coefficient: %.2f, LL: %.5f, Kinematic R2: %.5f ' % (elbo, cc, LL, kinematic_r2))
        else:
            print(f'Partition: {partition_use} Elbo: %.2f,Correlation coefficient: %.2f, LL: %.5f ' % (elbo, cc, LL))
        results[partition_use].update({'metrics': metrics})
            
        
        
    return results

# def posterior_sample_avg(model, neural, mask=None, n_samples=50, bias_scale=1):
#     # posterior sample and average
#     inp_samples = []
#     ic_samples = []
#     fr_samples = []
#     factor_samples = []

#     for iiter in range(n_samples):
#         if mask is None:
#             op = model.get_factors_ic(neural,
#                                     np.ones((neural.shape[0], 1)),
#                                     get_inputs=True, training=False, 
#                                     bias_scale=bias_scale)
#         else:
#             op = model.get_factors_ic(neural, mask,
#                                       np.ones((neural.shape[0], 1)),
#                                       get_inputs=True, training=False, 
#                                       bias_scale=bias_scale)
                                    
#         factors, ic_sample, firing_rate_prediction, factors_stacked, inp_sample = op
#         inp_samples += [inp_sample]
#         ic_samples += [ic_sample]
#         fr_samples += [firing_rate_prediction]
#         factor_samples += [factors]

#     inp_samples = np.array(inp_samples)
#     inp_samples = inp_samples.mean(0)

#     ic_samples = np.array(ic_samples)
#     ic_samples = ic_samples.mean(0)

#     fr_samples = np.array(fr_samples)
#     fr_samples = fr_samples.mean(0)

#     factor_samples = np.array(factor_samples)
#     factor_samples = factor_samples.mean(0)

#     return inp_samples, ic_samples, fr_samples, factor_samples


# # no bias input
# def posterior_sample_avg_noinp(model, neural, n_samples=50):
#     # posterior sample and average
#     ic_samples = []
#     fr_samples = []
#     factor_samples = []

#     for iiter in range(n_samples):
#         #         print(iiter)
#         op = model.get_factors_ic(neural,
#                                   np.ones((neural.shape[0], 1)),
#                                   training=False)
#         factors, ic_sample, firing_rate_prediction, factors_stacked = op
#         ic_samples += [ic_sample]
#         fr_samples += [firing_rate_prediction]
#         factor_samples += [factors]

#     ic_samples = np.array(ic_samples)
#     ic_samples = ic_samples.mean(0)

#     fr_samples = np.array(fr_samples)
#     fr_samples = fr_samples.mean(0)

#     factor_samples = np.array(factor_samples)
#     factor_samples = factor_samples.mean(0)

#     return ic_samples, fr_samples, factor_samples


# utils for fixed point finding
def find_fixed_point(generator, initial_state, static_input,
                     lr=0.1,
                     max_steps=25000, eps_stop=1e-7,
                     decay_steps=1, decay=1.0):

    n_points = initial_state.astype(np.float32).shape[0]
    state = tf.Variable(initial_state.astype(np.float32))

    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    loss_log = []
    lr_log = []
    for _ in tqdm(range(max_steps)):

        lr_log += [optimizer.lr]
        with tf.GradientTape() as tape:

            updated_state = generator(np.expand_dims(static_input, 1), # add extra time dimension
                                      initial_state = state)
            updated_state = updated_state[:, 0, :]
            loss_sample = tf.reduce_mean((state - updated_state )**2, axis=1)
            # loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean((state - updated_state )**2, axis=1)))
            loss = tf.reduce_mean((state - updated_state )**2)
            
            # temp = 10
            # loss = tf.math.reduce_logsumexp(tf.reduce_mean((state - updated_state )**2, axis=1) / temp, axis=0) * temp / n_points
            gradients = tape.gradient(loss, [state])
            optimizer.apply_gradients(zip(gradients, [state]))
            if np.abs(loss.numpy()) < eps_stop:
                break

            loss_log += [loss.numpy()]

    # plt.figure()
    # plt.semilogy(loss_log)
    # plt.title(eps_stop)
    # plt.show()

    return state.numpy(), loss_sample.numpy()


from sklearn import linear_model
def estimate_ic_for_lds_approx(state_traj, x_fp, J, sparsity=None, n_nonzero_coefs=None, reg_ls=None):
    L = state_traj.shape[0]

    A_list = []
    state_flat = []
    A = np.eye(J.shape[0])
    for il in range(L):
        A = A * J
        A_list += [A]
        state_flat += [state_traj[il, :] - x_fp]

    A_list = np.concatenate(A_list, axis=0)
    state_flat = np.concatenate(state_flat, axis=0)
    if sparsity is None and n_nonzero_coefs is None and reg_ls is None:
        x_init = np.linalg.pinv(A_list).dot(state_flat) + x_fp
    elif reg_ls is not None:
        x_init = np.linalg.inv(A_list.T.dot(A_list) + reg_ls * np.eye(A_list.shape[1])).dot(A_list.T.dot(state_flat)) + x_fp
    elif sparsity is not None:
        clf = linear_model.Lasso(alpha=sparsity)
        clf.fit(A_list, state_flat)
        x_init = clf.coef_ + x_fp
    else:
        reg = linear_model.Lars(n_nonzero_coefs=n_nonzero_coefs)
        reg.fit(A_list, state_flat)
        x_init = reg.coef_ + x_fp

    return x_init


def get_state_trajectories(model, neural_sample, session_id, training=False):

    op = model.get_factors_ic(neural_sample, session_id, get_inputs=True)
    factors, ic_sample, firing_rate_prediction, factors_stacked, inp_sample = op

    # neural encoding

    inp_time = tf.expand_dims(0 * neural_sample[:, :, 0], 2) +tf.expand_dims(inp_sample, 1)
    print(inp_time.shape, ic_sample.shape)
    state_traj = model.generators[0](tf.convert_to_tensor(inp_time),
                                     initial_state=model.ic_to_state[0](ic_sample))
    return state_traj.numpy()


def plot_states_fps_all_trials(state, fixed_points, colors_state, colors_fp, 
                                projection, ax,
                               alpha_state=1, alpha_fp=1, 
                               plot_type='3d', fp_marker='.', 
                               fp_markersize=10, state_lw=1):

    for iis, (s, fp, col_s, col_fp) in enumerate(zip(state,
                                           fixed_points,
                                           colors_state, colors_fp )):

        s_2d = (s - projection['mn']).dot(projection['q'])
        fp_2d = (fp - projection['mn']).dot(projection['q'])

        if plot_type == '3d':
            ax.plot3D(s_2d[:, 0], s_2d[:, 1], s_2d[:, 2], color=col_s, alpha=alpha_state, lw=state_lw)
            #ax.plot3D([s_2d[0, 0]], [s_2d[0, 1]], [s_2d[0, 2]], '.', color=col_s, alpha=alpha_state, markersize=fp_markersize)
        elif plot_type == '2d':
            ax.plot(s_2d[:, 0], s_2d[:, 1], color=col_s, alpha=alpha_state, lw=state_lw)
            ax.plot([s_2d[0, 0]], [s_2d[0, 1]], '.', color=col_s, alpha=alpha_state)

        for iifp in range(fp_2d.shape[0]):
            if plot_type == '3d':
                ax.plot3D(np.expand_dims(fp_2d[iifp, 0], 0),
                          np.expand_dims(fp_2d[iifp, 1], 0),
                          np.expand_dims(fp_2d[iifp, 2], 0),
                          fp_marker, color=col_fp, markersize=fp_markersize, alpha=alpha_fp)
            elif plot_type == '2d':
                ax.plot(np.expand_dims(fp_2d[iifp, 0], 0),
                        np.expand_dims(fp_2d[iifp, 1], 0),
                        fp_marker, color=col_fp, markersize=fp_markersize, alpha=alpha_fp)
    ax.grid(False)


def plot_temporal_features(feature, colors, projection, ax,
                           alpha=1, plot_type='3d'):

    for iis, (s, col) in enumerate(zip(feature,
                                       colors )):

        s_2d = (s - projection['mn']).dot(projection['q'])

        if plot_type == '3d':
            ax.plot3D(s_2d[:, 0], s_2d[:, 1], s_2d[:, 2], color=col, alpha=alpha)
        elif plot_type == '2d':
            ax.plot(s_2d[:, 0], s_2d[:, 1], color=col, alpha=alpha)


# Time warping axis
def time_warping_input(generator, states, inputs,
                       lr=0.01,
                       max_steps=250000, eps_stop=1e-7,
                       decay_steps=1, decay=1.0, speedup=1):
    du_init = np.random.randn(inputs.shape[-1]).astype(np.float32)
    du = tf.Variable(du_init)

    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    x = tf.constant(states.astype(np.float32))

    loss_log = []
    lr_log = []
    for istep in tqdm(range(max_steps)):

        lr_log += [optimizer.lr]
        with tf.GradientTape() as tape:

            x1u = generator(tf.expand_dims(inputs, 1),  # add extra time dimension
                            initial_state=x)
            x1u = x1u[:, 0, :]

            x1udu = generator(tf.expand_dims(inputs + du, 1),  # add extra time dimension
                              initial_state=x)
            x1udu = x1udu[:, 0, :]

            loss = tf.reduce_mean(((x1udu - x) - (speedup) * (x1u - x)) ** 2)

            gradients = tape.gradient(loss, [du])
            optimizer.apply_gradients(zip(gradients, [du]))

            loss_log += [loss.numpy()]

            if loss_log[-1] < eps_stop:
                break

    plt.semilogy(loss_log)
    plt.title(eps_stop)
    plt.show()

    # speed direction 
    
    return du.numpy()


# Fixed points
def keep_unique_fixed_points(losses, fps, identical_tol=0.0, do_print=True):
    """Get unique fixed points by choosing a representative within tolerance.
    Args:
      losses: speeds
      fps: numpy array, FxN tensor of F fixed points of N dimension
      identical_tol: float, tolerance for determination of identical fixed points
      do_print: Print useful information?
    Returns:
      2-tuple of UxN numpy array of U unique fixed points and the kept indices
    """
    keep_idxs = onp.arange(fps.shape[0])
    if identical_tol <= 0.0:
        return losses, fps #, keep_idxs
    if fps.shape[0] <= 1:
        return losses, fps # , keep_idxs

    nfps = fps.shape[0]
    example_idxs = onp.arange(nfps)
    all_drop_idxs = []

    # If point a and point b are within identical_tol of each other, and the
    # a is first in the list, we keep a.
    distances = squareform(pdist(fps, metric="euclidean"))
    for fidx in range(nfps - 1):
        distances_f = distances[fidx, fidx + 1:]
        drop_idxs = example_idxs[fidx + 1:][distances_f <= identical_tol]
        all_drop_idxs += list(drop_idxs)

    unique_dropidxs = onp.unique(all_drop_idxs)
    keep_idxs = onp.setdiff1d(example_idxs, unique_dropidxs)
    if keep_idxs.shape[0] > 0:
        unique_fps = fps[keep_idxs, :]
    else:
        unique_fps = onp.array([], dtype=onp.int64)

    if do_print:
        print("    Kept %d/%d unique fixed points with uniqueness tolerance %f." %
              (unique_fps.shape[0], nfps, identical_tol))

    return losses[keep_idxs], unique_fps


def fixed_points_with_tolerance(losses, fps, tol=onp.inf, do_print=True):
    """Return fixed points with a fixed point loss under a given tolerance.

    Arguments:
      rnn_fun: one-step update function as a function of hidden state
      fps: ndarray with shape npoints x ndims
      tols: loss tolerance over which fixed points are excluded
      do_print: Print useful information?
    Returns:
      2-tuple of kept fixed points, along with indicies of kept fixed points
    """
    lidxs = losses < tol
    keep_idxs = onp.where(lidxs)[0]
    fps_w_tol = fps[lidxs]
    losses_select = losses[lidxs]
    
    if do_print:
        print("    Kept %d/%d fixed points with tolerance under %f." %
              (fps_w_tol.shape[0], fps.shape[0], tol))

    return losses_select, fps_w_tol 


def exclude_outliers(losses, data, outlier_dist=onp.inf, metric='euclidean', do_print=True):
    """Exclude points whose closest neighbor is further than threshold.
    Args:
      data: ndarray, matrix holding datapoints (num_points x num_features).
      outlier_dist: float, distance to determine outliers.
      metric: str or function, distance metric passed to scipy.spatial.pdist.
          Defaults to "euclidean"
      do_print: Print useful information?
    Returns:
      2-tuple of (filtered_data: ndarray, matrix holding subset of datapoints,
        keep_idx: ndarray, vector of bools holding indices of kept datapoints).
    """
    if onp.isinf(outlier_dist):
        return losses, data #, onp.arange(len(data))
    if data.shape[0] <= 1:
        return losses, data #, onp.arange(len(data))

    # Compute pairwise distances between all fixed points.
    distances = squareform(pdist(data, metric=metric))

    # Find second smallest element in each column of the pairwise distance matrix.
    # This corresponds to the closest neighbor for each fixed point.
    closest_neighbor = onp.partition(distances, 1, axis=0)[1]

    # Return data with outliers removed and indices of kept datapoints.
    keep_idx = onp.where(closest_neighbor < outlier_dist)[0]
    data_to_keep = data[keep_idx]

    if do_print:
        print("    Kept %d/%d fixed points with within outlier tolerance %f." %
              (data_to_keep.shape[0], data.shape[0], outlier_dist))

    return losses[keep_idx], data_to_keep #, keep_idx


# NOTE - 2023-01-08: Make sure this is correct for a function for which I know answer analytically
def get_jacobian(generator, state_, inp_):
    '''
    Args:
    state_ : hidden
    inp_ : hidden
    '''
    J = []
    for iss in range(state_.shape[0]):
        with tf.GradientTape() as tape:
            static_input = np.expand_dims(inp_, 0).astype(np.float32)
            state = tf.Variable(state_.astype(np.float32))
            updated_state = generator(np.expand_dims(static_input, 1),  # add extra time dimension
                                      initial_state=tf.expand_dims(state, 0))
            updated_state = updated_state[0, 0, :]
            #             state_diff = updated_state - state
            J += [tape.gradient(updated_state[iss], [state])[0]]

    return [j.numpy() for j in J]


def get_alignment(L, v, imag_component=1e-4):
    # L.T.dot(v); columns correspond to different eigen vectors -- taken from David Z

    from scipy.linalg import subspace_angles

    angles = []
    for il in range(L.shape[1]):
        l = L[:, il: il + 1]
        imag_component = np.linalg.norm(np.imag(l))
        if imag_component > 1e-4:
            v1 = np.hstack((np.real(l), np.imag(l)))
        else:
            v1 = np.real(l)

        ang = np.min(subspace_angles(v, v1))
        angles.append(ang)

    return np.cos(np.array(angles))


def plot_eigen_vals(Jev_log, color_log, ax, sz=None,
                    alpha=0.5, s=1, c='k', 
                    percentile_lower=5, percentile_higher=95, cmap='bwr', 
                    dt=1, plot_type='color', vmin=None, vmax=None):
    pts_log = []
    alpha_log = []
    sz_log = []
    for iis in range(len(Jev_log)):
        Jev_sample_log = Jev_log[iis]
        for ijev, jev in enumerate(Jev_sample_log):
            pts_log += [jev['lam']]
            alpha_log += [color_log[iis][ijev]]
            if sz is not None:
                sz_log += [sz[iis][ijev]]
            

    pts_log = np.concatenate(pts_log, axis=0)
    alpha_log = np.concatenate(alpha_log, axis=0)
    
    xval = np.abs(pts_log)
    # yval =  np.angle(pts_log)  / (2 * np.pi * dt)

    # xval = np.log(np.abs(pts_log))
    yval = np.angle(pts_log)  / (2 * np.pi * dt)

    if plot_type == 'color':
        ax.scatter(xval, yval, s=s, c=alpha_log, alpha=alpha,
                vmin=np.percentile(alpha_log, percentile_lower), 
                vmax=np.percentile(alpha_log, percentile_higher), 
                cmap=cmap)
    elif plot_type == 'size':
        ax.scatter(xval, yval, s=alpha_log, c=c, alpha=alpha)
    elif plot_type == 'color_size':
        ax.scatter(xval, yval, s=sz_log, c=alpha_log, alpha=alpha, cmap=cmap, 
                    vmin=vmin, vmax=vmax)
        # 
    #dt / (-np.log(np.abs(pts_log)))
    # ax.set_xlabel(r'|log($\lambda$|)')
    ax.set_xlabel(r'|$\lambda$|', fontsize=10)
    # ax.set_xlabel('Decay (s)')
    ax.set_ylabel(r'Hz', fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.gca().invert_xaxis()


#     ax.plot(np.abs(pts_log), np.angle(pts_log), '.')
#     ax.colorbar()

def evolve_linear_dynamics(x0, A, fp, time_steps):
    x = [x0]
    xt = x0
    for _ in range(time_steps):
        xt = A.dot(xt - fp) + fp
        x += [xt]

    x = np.array(x)
    return x


def evolve_linear_dynamics_with_reset(states, A, fp, time_steps):
    x = []
    for itime in range(time_steps):
        xt = A.dot(states[itime, :] - fp) + fp
        x += [xt]

    x = np.array(x)
    return x


def find_fixed_pts_contextual_lfads(model, inputs, states, noise_state=0.1, n_inits_per_example=20,
                                    eps_list=[1e-10], max_steps=60000, lr=1, decay=0.9996, decay_steps=10):


    NOISE_STATE = noise_state
    BATCH_SZ = n_inits_per_example

    inp_log = []
    state_init_log = []
    state_tm_log = []
    n_examples = inputs.shape[0]


    for iex in range(n_examples):
        inp_ex = inputs[iex, ...]
        state_traj = states[iex, ...]

        state_inits = []
        inp_log_ = []
        for _ in range(BATCH_SZ):
            inp_log_ += [inp_ex] #[np.squeeze(inp_ex)]
            t_choice = np.random.choice(np.arange(state_traj.shape[0]), 1)[0]
            state_init = state_traj[t_choice, :] + np.random.randn(state_traj.shape[-1]) * NOISE_STATE
            state_inits += [state_init]

        state_init_log += [state_inits]
        inp_log += [np.array(inp_log_)]
        state_tm_log += [state_traj]

    import time

    fp = []
    loss_sample = []

    # plt.figure(figsize=(5 * len(eps_list), 5))
    for ieps, eps in enumerate(eps_list):
        # plt.subplot(1, len(eps_list), ieps + 1)
        t_s = time.time()
        fp_, loss_sample_ = find_fixed_point(model.generators[0],
                                             np.concatenate(state_init_log, axis=0),
                                             # Since the state changes a lot from the IC, use samples from the trajectory for FP finding.
                                             np.concatenate(inp_log, axis=0),
                                             eps_stop=eps, lr=np.float32(lr),  # lr=0.1
                                             decay= decay,
                                             decay_steps=decay_steps,
                                             max_steps=max_steps)
        # plt.show()
        fp += [fp_]
        loss_sample += [loss_sample_]

        print(f'fixed point finding time: {time.time() - t_s}')
    

    fp = np.concatenate(fp, axis=0)
    loss_sample = np.concatenate(loss_sample, axis=0)

    fp_log = [fp[i * BATCH_SZ * len(eps_list): (i + 1) * BATCH_SZ * len(eps_list)] for i in range(n_examples)]
    loss_sample_log = [loss_sample[i * BATCH_SZ * len(eps_list): (i + 1) * BATCH_SZ * len(eps_list)] for i in
                       range(n_examples)]

    return fp_log, loss_sample_log

def find_jacobians_contextual_lfads(model, fp_log__, inp_log_all):

    Jev_log = []
    for ipt in tqdm(range(len(fp_log__))):
        Jev_sample_log = []
        for iipt in range(fp_log__[ipt].shape[0]):
            J = get_jacobian(model.generators[0],
                             fp_log__[ipt][iipt, :], inp_log_all[ipt, :])
            J = np.array(J)
            lam, right_eigenvec = np.linalg.eig(J)  # J = EVE-1 = RVL.
            left_eigenvec = np.linalg.inv(right_eigenvec)  ## NOTE 04/20/23: Use pinv instead of inv?

            Jev_sample_log += [{'J': J, 'right_eigenvec': right_eigenvec,
                                'left_eigenvec': left_eigenvec, 'lam': lam}]

        Jev_log += [Jev_sample_log]
    return Jev_log


def find_ics_for_linearized_dynamics(fp_log__, state_tm_log, Jev_log, tstart=0, reg_ls=None, 
                                     sparsity=None,  n_nonzero_coefs=None):
    x_init_log = []
    for iis in tqdm(range(len(Jev_log))):
        Jev_sample_log = Jev_log[iis]
        x_init_sample_log = []
        for ijev, jev in enumerate(Jev_sample_log):
            x_fixed_pt = fp_log__[iis][ijev, :]
            x_init = estimate_ic_for_lds_approx(state_tm_log[iis][tstart:, :],
                                                x_fixed_pt,
                                                jev['J'],
                                                sparsity=sparsity,
                                                n_nonzero_coefs=n_nonzero_coefs, 
                                                reg_ls=reg_ls)
            x_init_sample_log += [x_init]

        x_init_log += [x_init_sample_log]

    return x_init_log


from sklearn import linear_model
def estimate_J_for_lds_approx(state_traj, x_fp, ridge_alpha=0):
    L = state_traj.shape[0]
    

    state_t = []
    state_tp1 = []
    for il in range(L-1):
        state_t += [state_traj[il, :] - x_fp]
        state_tp1 += [state_traj[il+1, :] - x_fp]

    state_t = np.array(state_t)
    state_tp1 = np.array(state_tp1)

    # need J, such that state_tp1 = J * state_t
    # state_tp1' = state_t'J'
    # (state_t')-1 state_tp1' = J'
    from sklearn.linear_model import LinearRegression, Ridge
    # reg = LinearRegression().fit(state_t, state_tp1)
    reg = Ridge(alpha=ridge_alpha).fit(state_t, state_tp1 - state_t)
    Jt = reg.coef_ 
    Jt = Jt + np.eye(Jt.shape[0])
    # J = Jt.T
    J = Jt
    return J

def find_J_for_linearized_dynamics(fp_log__, state_tm_log, pca=None, ridge_alpha=0):
    J_log = []
    for iis in tqdm(range(len(fp_log__))):
        J_sample_log = []
        for ijev, jev in enumerate(fp_log__[iis]):
            x_fixed_pt = fp_log__[iis][ijev, :]
            if pca is None:
                J = estimate_J_for_lds_approx(state_tm_log[iis],
                                                    x_fixed_pt )
            else:
                J = estimate_J_for_lds_approx(pca.transform(state_tm_log[iis]),
                                              pca.transform(np.expand_dims(x_fixed_pt, 0) )[0, :], 
                                              ridge_alpha=ridge_alpha)
            J = np.array(J)
            lam, right_eigenvec = np.linalg.eig(J)  # J = EVE-1 = RVL.
            left_eigenvec = np.linalg.inv(right_eigenvec)  ## NOTE 04/20/23: Use pinv instead of inv?

            J_sample_log += [{'J': J, 'right_eigenvec': right_eigenvec,
                                'left_eigenvec': left_eigenvec, 'lam': lam}]

        J_log += [J_sample_log]

    return J_log


import scipy
def smoothen(x, sigma=5):
    # time x dimensions

    xx = []
    for idim in range(x.shape[1]):
        xx += [scipy.ndimage.gaussian_filter(x[:, idim], sigma)]
    xx = np.array(xx)
    return xx.T

# visualizations
from sklearn.decomposition import PCA
def low_dim_analysis(cues, features,
                 start_tm=0, end_tm=np.inf, collapse_time=False,
                 n_comps_cis=3, smoothen_sig=0,
                 trial_ids=None, do_trial_avg=False,
                 n_comps_pca=2, method='pca', do_qr=True, 
                 add_cis_dimension=False):
    import copy

    if smoothen_sig > 0:
        feat = []
        for itrial in range(features.shape[0]):
            feat += [smoothen(features[itrial, ...], sigma=smoothen_sig)]
        features = np.array(feat)

    # remove mean
    feature_selected = copy.deepcopy(features)
    mn = np.reshape(feature_selected, [-1, feature_selected.shape[-1]]).mean(0)
    feature_selected_mean_removed = feature_selected - mn

    # optionally, remove trial dimension
    P = np.eye(mn.shape[0])
    if trial_ids is not None:
        trials_selected = trial_ids
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
    feature_selected = copy.deepcopy(features)
    feature_selected_mean_removed = feature_selected - mn
    feature_selected_mean_removed = np.einsum('ijk,kl->ijl', feature_selected_mean_removed, P)
    cis = np.nanmean(feature_selected_mean_removed, 0)
    u, s, v = np.linalg.svd(cis.T)
    q_cis = u[:, 0]
    P_cis_orth = np.eye(u.shape[0]) - u[:, :n_comps_cis].dot(u[:, :n_comps_cis].T)
    P = P.dot(P_cis_orth)

    # learn PCA dims
    if method == 'pca':
        feature_selected = copy.deepcopy(features)
        feature_sel_ = feature_selected

        if do_trial_avg:
            unique_cues = np.unique(cues, axis=0)
            features_tr_avg = []
            cues_tr_avg = []
            for icue in range(unique_cues.shape[0]):
                ic = np.where(np.sum(np.abs(unique_cues[icue, :] - cues), 1) == 0)[0]
                features_tr_avg += [feature_sel_[ic, ...].mean(0)]
                cues_tr_avg += [unique_cues[icue]]

            feature_sel_ = np.array(features_tr_avg)

        if collapse_time:
            feature_sel_ = feature_sel_[:, start_tm: end_tm, :].mean(-2)
        else:
            feature_sel_ = np.reshape(feature_sel_, [-1, feature_sel_.shape[-1]])

        feature_sel_ -= mn
        feature_sel_ = feature_sel_.dot(P)

        pca = PCA(n_components=n_comps_pca)
        pca.fit(feature_sel_)
        q, _ = np.linalg.qr(pca.components_.T)
        q = P_cis_orth.dot(q)

    if method == 'mpca':

        feature_sel_ = copy.deepcopy(feature_selected)
        feature_sel_ = feature_sel_ - mn
        feature_sel_2 = feature_sel_.dot(P.T)
        q_collect = []
        n_elements = len(cues[0])
        for ielem in range(n_elements):

            cue_position = [c[ielem] for c in cues]
            unique_cues = set(cue_position)
            cond_avg = []
            for icue_pos in unique_cues:
                print(icue_pos)
                selected_trials = [i for i in range(len(cue_position)) if cue_position[i] == icue_pos]
                print(ielem, icue_pos, selected_trials, len(selected_trials))
                cond_avg += [feature_sel_2[selected_trials, ...].mean(0)]

            cond_avg = np.concatenate(cond_avg, axis=0)
            print('cond_avg', cond_avg.shape)

            pca = PCA(n_components=10)
            pca.fit(cond_avg)
            plt.plot(np.cumsum(pca.explained_variance_ratio_), '-*')
            #             u, s, v = pca.fit(cond_avg.T)
            #             plt.plot(s**2 / np.sum(s**2), '-*')
            q_collect += [pca.components_[0, :]]

            plt.show()

        q_collect = np.array(q_collect).T
        print('q_collect', q_collect.shape)
        if do_qr:
            q, _ = np.linalg.qr(q_collect)
        else:
            q = q_collect

    if add_cis_dimension:
        q = np.concatenate([np.expand_dims(q_cis, 1), q], axis=1)
        if do_qr:
            q, _ = np.linalg.qr(q)

    features = feature_selected
    return mn, q, features, cues


def plot_low_dim(feature_selected, kinematics, mn, q, new_fig=True, ls='-', ms='.', cols=None):
    unique_kin = np.unique(kinematics, axis=0)

    if cols is None:
        rng = np.random.default_rng(55)
        cols = rng.uniform(size=(unique_kin.shape[0], 3))  # NOTE 03/31/23: HSV interpolation.

    if new_fig:
        plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    for itrial in range(feature_selected.shape[0]):
        feat2d = (feature_selected[itrial, :, :] - mn).dot(q)

        ic = cols[np.argmin(np.sum(np.abs(unique_kin - kinematics[itrial, :]), 1)), :]
        plt.plot(feat2d[:, 0], feat2d[:, 1], color=ic, ls=ls)
        plt.plot(feat2d[0, 0], feat2d[0, 1], ms, markersize=30, color=ic)

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
        plt.plot(feat_avg_traj[:, 0], feat_avg_traj[:, 1], linewidth=3, color=cols[ikin], ls=ls)
        plt.plot(feat_avg_traj[0, 0], feat_avg_traj[0, 1], ms, markersize=30, color=cols[ikin])

    plt.axis('equal')
    plt.title('trial average')


def get_trial_avgs(spikes_trials_jpca, label_trials_jpca, smoothen_sig=-1, subsample=-1):
    # do trial averaging
    
    spikes_trial_averaged = []
    n_trials = []

    trial_labels = list(set(label_trials_jpca))
    for ref_trial in trial_labels:
        trials_selected = [itrial for itrial in range(len(spikes_trials_jpca)) if label_trials_jpca[itrial] == ref_trial]
        spikes_selected = spikes_trials_jpca[trials_selected, ...]
        if smoothen_sig > 0:
            spikes_selected = np.array([smoothen(spikes_selected[i, :, :], sigma=smoothen_sig) for i in range(spikes_selected.shape[0])])
        spikes_selected = np.nanmean(spikes_selected, 0)
        spikes_trial_averaged += [spikes_selected]
        n_trials += [len(trials_selected)]
    
    spikes_trial_averaged = np.array(spikes_trial_averaged)
    
    if subsample > 0:
        bin_w = subsample
        spikes_trial_averaged_binned = np.zeros((spikes_trial_averaged.shape[0], 
                                                 int(spikes_trial_averaged.shape[1] / bin_w), 
                                                 spikes_trial_averaged.shape[2]))

        for ibin in range(spikes_trial_averaged_binned.shape[1]):
            spikes_trial_averaged_binned[:, ibin, :] = spikes_trial_averaged[:, ibin*bin_w: (ibin+1)*bin_w, :].sum(1)

        spikes_trial_averaged = spikes_trial_averaged_binned
        
    return spikes_trial_averaged, trial_labels, n_trials



def get_firing_rate_error(model, states, fr, A, fp, session_id):
    """
    Calculate the mean squared error between the natural logarithm of actual firing rates
    and predicted firing rates from a linear dynamical system model with a non-linear transformation.

    Parameters:
    states (ndarray): Array of neural states.
    fr (ndarray): Array of actual firing rates.
    A (ndarray): State transition matrix of the linear dynamical system.
    fp (float): Reset condition parameter for the dynamical system.
    session_id (int): Session ID for the output layer.

    Returns:
    float: The computed mean squared error.
    """

    

    # Determine the number of time steps from the shape of the states array
    time_steps = states.shape[0]

    # Evolve the state dynamics over the given time steps
    linear_state_dynamics = evolve_linear_dynamics_with_reset(states, A, fp, time_steps)

    # Apply a non-linear transformation to the linear state dynamics
    # print('session_id', session_id)
    states_updated = model.factor_mappings[0](linear_state_dynamics)
    states_updated = tf.expand_dims(states_updated, 0)
    # print('states_updated', states_updated)
    linearly_predicted_logfr = model.factor_op_mapping((session_id, states_updated))[0, ...]
    # print('linearly_predicted_logfr', linearly_predicted_logfr)

    # Compute the mean squared error between the log of actual firing rates and predicted log firing rates
    mse = np.mean((np.log(fr) - linearly_predicted_logfr)**2)
    kl = np.mean(fr * (np.log(fr) - linearly_predicted_logfr) + np.exp(linearly_predicted_logfr) - fr)
    
    return mse, kl

def compute_mode_errors(model, jac_details, states, fr, fp, session_id):
    """
    Compute the mean squared error for different configurations of a Jacobian matrix
    by removing its modes (eigenvalues) one at a time.

    Parameters:
    jac_details (dict): Dictionary containing eigenvalues ('lam') and eigenvectors 
                        of a Jacobian matrix.
    states (ndarray): Array of neural states.
    fr (ndarray): Array of actual firing rates.
    fp (float): Reset condition parameter for the dynamical system.
    session_id (int): Session ID for the output layer.

    Returns:
    ndarray: An array of MSE values, each corresponding to a configuration with a mode removed.
    """
    mse_log_mode_removed = []
    kl_log_mode_removed = []

    # Iterate over each mode (eigenvalue)
    for imode_removed in range(jac_details['lam'].shape[0]):
        # Extract the eigenvalue and its complex conjugate
        lam_removed = jac_details['lam'][imode_removed]
        lam_conj = np.conj(lam_removed)

        # Generate an array of indices, remove the current mode and its conjugate
        select = np.arange(jac_details['lam'].shape[0])
        select = np.delete(select, imode_removed)
        select = np.delete(select, np.where(jac_details['lam'][select] == lam_conj))

        # Reconstruct A matrix without the selected modes
        A_recons = np.real((jac_details['right_eigenvec'][:, select] * jac_details['lam'][select])
                           .dot(jac_details['left_eigenvec'][select, :]))

        # Calculate MSE for the reconstructed system
        mse_, kl_ =  get_firing_rate_error(model, states, fr, A_recons, fp, session_id)

        # Append MSE to the log
        mse_log_mode_removed += [mse_]
        kl_log_mode_removed += [kl_]

    # Convert MSE log to a numpy array
    mse_log_mode_removed = np.array(mse_log_mode_removed)
    kl_log_mode_removed = np.array(kl_log_mode_removed)

    return mse_log_mode_removed, kl_log_mode_removed

def analyse_mode_importance(model, session_id, firing_rate, state, fixed_points_filtered, jacobians):
    """
    Analyze the importance of different modes (eigenvalues) of Jacobian matrices
    in predicting neural firing rates for multiple examples.

    Parameters:
    model (object): The contextual LFADS model. 
    session_id : session ID for output layers
    firing_rate (ndarray): Array of actual firing rates for multiple examples.
    state (ndarray): Array of neural states for multiple examples.
    fixed_points_filtered (list): List of fixed points for each example.
    jacobians (list): List of Jacobian matrices and their details for each example.

    Returns:
    list: A list of dictionaries for each example, containing the MSEs for different
          configurations of Jacobian matrices.
    """
    mode_errors = []

    # Iterate over each example
    for iexample in tqdm(range(firing_rate.shape[0])):
        fr = firing_rate[iexample, ...]
        states = state[iexample, ...]

        # Check if there are no fixed points and continue to the next example if so
        if len(fixed_points_filtered[iexample]) == 0:
            mode_errors += [[]]
            continue

        mse_fps = []
        # Iterate over each fixed point for the current example
        for ifp in range(len(fixed_points_filtered[iexample])):
            fp = fixed_points_filtered[iexample][ifp]
            jac_details = jacobians[iexample][ifp]
            A = jac_details['J']

            # Compute MSE with all modes
            mse_fullA, kl_fullA = get_firing_rate_error(model, states, fr, A, fp, session_id)

            # Compute MSE with individual modes removed
            mse_log_mode_removed, kl_log_mode_removed = compute_mode_errors(model, jac_details, states, fr, fp, session_id)

            # Store the MSEs in a dictionary
            mse_fps += [{'mse_allmodes': mse_fullA, 'kl_allmodes': kl_fullA, 
                        'mse_log_mode_removed': mse_log_mode_removed, 'kl_log_mode_removed': kl_log_mode_removed}]

        mode_errors += [mse_fps]
    
    return mode_errors


def get_nearest_mode(movement_angle, jacobians, speeds, states, fixed_pts, eval_ref, plot_on=False):
    """
    Identifies the nearest mode of movement based on eigenvalues and plots if required.

    Parameters:
    - cues: An array of cues indicating movement directions.
    - jacobians: A list of jacobian matrices for different points in time.
    - speeds: An array of speeds corresponding to different modes.
    - states: An array of states.
    - eval_ref: The reference eigenvalue for comparison.
    - plot_on: Boolean flag to enable plotting.

    Returns:
    - e_mode_log: Log of eigenmodes closest to the reference eigenvalue.
    - movement_angle_log: Log of movement angles corresponding to the eigenmodes.
    - states_log: Log of states corresponding to the eigenmodes.
    """

    # Initialize logs for movement angle, eigenmodes, and states
    movement_angle_log = []
    e_mode_log = []
    dynamics_mode_log = []
    states_log = []
    
    # Iterate through each point to find and log the nearest mode
    for ipt in range(len(jacobians)):
        
        # Skip if there are no jacobians for this point
        if len(jacobians[ipt]) == 0:
            continue
        
        # Find the index of the slowest mode based on speed
        iipt = np.argmin(speeds[ipt])
        
        # Find the mode with eigenvalue closest to the reference
        imode = np.argmin(np.abs(jacobians[ipt][iipt]['lam'] - eval_ref))
        jmode = np.argmin(np.abs(jacobians[ipt][iipt]['lam'] - np.conj(eval_ref)))
        
        # Extract the corresponding right eigenvector and form the plane
        evec = jacobians[ipt][iipt]['right_eigenvec'][:, imode]
        plane = [np.real(evec), np.imag(evec)]
        plane = np.array(plane)

        # get selected dynamics
        R = jacobians[ipt][iipt]['right_eigenvec'][:, np.array([imode, jmode])]
        Lams = np.diag(jacobians[ipt][iipt]['lam'][np.array([imode, jmode])])
        L = jacobians[ipt][iipt]['left_eigenvec'][np.array([imode, jmode]), :]
        A_selected = R.dot(Lams.dot(L))
        fixed_pt = fixed_pts[ipt][iipt]


        # Log the plane (eigenmode), movement angle, and state
        e_mode_log += [plane.T]
        dynamics_mode_log += [{'A': A_selected, 'fp': fixed_pt}]
        movement_angle_log += [movement_angle[ipt]]
        states_log += [states[ipt, ...]]

        # Plotting, if enabled
        if plot_on:
            idx = np.arange(jacobians[ipt][iipt]['lam'].shape[0])
            idx = np.setdiff1d(idx, imode)
            plt.plot(np.abs(jacobians[ipt][iipt]['lam'][idx]), 
                     np.angle(jacobians[ipt][iipt]['lam'][idx]), 'k.', markersize=0.1)
            plt.plot(np.abs(jacobians[ipt][iipt]['lam'][imode]), 
                     np.angle(jacobians[ipt][iipt]['lam'][imode]), 'r.', markersize=0.5)
            plt.ylim([-0.5, 0.5])
     
    # Final plotting adjustments, if plotting was done
    if plot_on:
        plt.plot(np.abs(eval_ref), np.angle(eval_ref), 'g+')
        
    # Convert logs to numpy arrays for return
    e_mode_log, movement_angle_log, states_log = np.array(e_mode_log), np.array(movement_angle_log), np.array(states_log)
    return e_mode_log, dynamics_mode_log, movement_angle_log, states_log

def pass_through_to_output(model, e_mode_log):
    """
    Passes eigenmodes through a model to get output eigenmodes.

    Parameters:
    - model: The model through which eigenmodes are passed.
    - e_mode_log: Log of eigenmodes to be passed through the model.

    Returns:
    - e_mode_log: Transformed log of eigenmodes after passing through the model.
    """
    
    # Apply factor operation mapping for each dimension of the eigenmode
    if len(e_mode_log.shape) == 3:
        e_mode_log_ = []
        for imode in range(e_mode_log.shape[-1]):
            e_mode_log_ += [model.factor_op_mapping(model.factor_mappings[0](e_mode_log[:, :, imode]))]
        e_mode_log_ = np.transpose(np.array(e_mode_log_), [1, 2, 0])
    else:
        e_mode_log_ = [model.factor_op_mapping(model.factor_mappings[0](e_mode_log[:, :]))]
        e_mode_log_ = np.array(e_mode_log_)

    return e_mode_log_
    

def measure_subspace_similarity(e_mode_log, movement_angle_log):
    """
    Measures the similarity between subspaces defined by eigenmodes.

    Parameters:
    - e_mode_log: Log of eigenmodes whose subspaces' similarity is to be measured.

    Returns:
    - angles_mat: A matrix of angles measuring subspace similarities.
    """
    # Sort the eigenmodes based on movement angle for comparison
    
    ifp_order = np.argsort(movement_angle_log)
    e_mode_log = e_mode_log[ifp_order, ...]
    movement_angle_log = movement_angle_log[ifp_order]
    from scipy.linalg import subspace_angles
    
    # Initialize matrix to hold subspace angles
    angles_mat = np.zeros((e_mode_log.shape[0], e_mode_log.shape[0], 2)) + np.nan
    
    # Compute subspace angles between each pair of eigenmodes
    for ipt in tqdm(range(e_mode_log.shape[0])):
        for jpt in range(e_mode_log.shape[0]):
            angles = subspace_angles(e_mode_log[ipt, ...], e_mode_log[jpt, ...])
            angles_mat[ipt, jpt, :] = angles
    
    # Convert angles from radians to degrees
    angles_mat = np.rad2deg(angles_mat)
    
    return angles_mat, movement_angle_log


###################### mPCA analysis ########################
def make_condition_groups(delays_trials, cues_trials):
    delays_trials_str = [str(i)[1:-1] for i in delays_trials]
    cues_trials_str = [str(i)[1:-1] for i in cues_trials]
    delays_trials_unique = set(delays_trials_str)

    condition_groups = {}
    for idelay_cond in delays_trials_unique:
        condition_groups.update({idelay_cond: {}})
        for itrial in range(len(delays_trials_str)):
            if delays_trials_str[itrial] == idelay_cond:
                if cues_trials_str[itrial] not in condition_groups[idelay_cond].keys():
                    condition_groups[idelay_cond].update({cues_trials_str[itrial]: []})
                condition_groups[idelay_cond][cues_trials_str[itrial]] += [itrial]

    return condition_groups


def mPCA(condition_groups, trial_ids, neural_trials, dt=5, new_fig=True, projections=None):
    n_delays = len(condition_groups.keys())

    n_cues_max = np.max([len(condition_groups[idelay_cond].keys()) for
                        idelay_cond in condition_groups.keys()])

    rng = np.random.default_rng(23)
    cols = rng.random((70, 3))
    print(n_delays, n_cues_max)
    projections_used = {}
    
    if new_fig:
        plt.figure(figsize=(5 * n_cues_max, 5 * n_delays))

    for iidel, idelay_cond in enumerate(condition_groups.keys()):
        # collect trials corresponding to conditions
        collect_tr_avgs = []
        collect_tr_ids = []
        for imove_cond in condition_groups[idelay_cond].keys():
            relevant_trials = np.intersect1d(trial_ids, condition_groups[idelay_cond][imove_cond])
            collect_tr_avgs += [neural_trials[relevant_trials, ...].mean(0)]
            collect_tr_ids += [imove_cond]

        # get max length for the delay
        seq_len = []
        for iicond, icond in enumerate(collect_tr_ids):
            seq_len += [len(str(icond).split(','))]
        n_elements = np.max(seq_len)

        # get CIS
        CIS = np.array(collect_tr_avgs).mean(0)
        
        if CIS.shape[-1] > 1:
            if projections is None:
                pca = PCA(n_components=1)
                CIS_1d = pca.fit_transform(CIS)
                projections_used.update({'CIS_projection':  pca})
            else:
                print('using given projections')
                CIS_1d = projections['CIS_projection'].fit_transform(CIS)
                projections_used.update({'CIS_projection':  projections['CIS_projection']})
                
        else:
            CIS_1d = CIS
        plt.subplot(n_delays, n_cues_max, n_cues_max * iidel + 1)
        plt.plot(np.arange(CIS_1d.shape[0]) * dt/1000, CIS_1d)
        plt.title(f'CIS {idelay_cond}')
        plt.xlabel('sec')

        # remove CIS
        collect_tr_avgs = [x - CIS for x in collect_tr_avgs]
        # n_elements = len(idelay_cond.split(','))
        for ilocation in range(n_elements):

            # get levels
            levels = []
            for iicond, icond in enumerate(collect_tr_ids):
                levels += [icond.split(',')[ilocation].strip()]
            levels = set(levels)

            tr_avgs_elem = []
            singletr_elem = []
            for ielem in levels:
                avg = []
                for iicond, icond in enumerate(collect_tr_ids):
                    if icond.split(',')[ilocation].strip() == ielem:
                        avg += [collect_tr_avgs[iicond]]
                avg_ = np.array(avg).mean(0)
                #print(idelay_cond, ilocation, ielem, np.array(avg).shape)
                tr_avgs_elem += [avg_]
                singletr_elem += [avg]
               
            if projections is None:
                pca = PCA(n_components=1)
                pca.fit(np.concatenate(tr_avgs_elem, axis=0))
            else:
                pca = projections[ilocation]
                print('using given projections')
            
            projections_used.update({ilocation: pca})
                
            plt.subplot(n_delays, n_cues_max, n_cues_max * iidel + 2 + ilocation)

            for icc, cc in enumerate(tr_avgs_elem):
                # print(icc)
                pcacc = pca.transform(cc)
                plt.plot(np.arange(pcacc.shape[0]) * dt/1000, pcacc, color=cols[icc])
                plt.xlabel('sec')

            # for icc, cc in enumerate(singletr_elem):
            #     for xcc in cc:
            #         plt.plot(pca.transform(xcc), color=cols[icc], alpha=0.1)

            plt.legend(levels, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
            
    return projections_used


def analysis_mpca(delays_trials, cues_trials, neural_trials, dt=5, new_fig=True, projections=None):
    condition_groups = make_condition_groups(delays_trials, cues_trials)
    projections_used = mPCA(condition_groups, np.arange(len(neural_trials)), np.array(neural_trials), dt=dt, new_fig=new_fig, projections=projections)
    return projections_used


def add_zero_bias(model, results):
    bias_dim = model.bias_dim
    for partition in ['train', 'eval', 'test']:
        nsamples = results[partition]['state'].shape[0]
        results[partition].update({'bias_sample': np.zeros((nsamples, bias_dim))})
        
    return results

def filter_fixedpoints(fixed_points, speed_fps, EPS1 = 1e-7, EPS2 = 1e-1, EPS3 = 1.0):

    fp_log__ = [fixed_points_with_tolerance(l, f, tol=EPS1, do_print=True)
               for l, f in zip(speed_fps, fixed_points)]
    fp_log__ = [keep_unique_fixed_points(l, f, identical_tol=EPS2) for l, f in fp_log__]
    fp_log__ = [exclude_outliers(l, f, outlier_dist=EPS3) for l, f in fp_log__]

    speeds_filtered = [l for l, f in fp_log__]
    fp_log__ = [f for l, f in fp_log__]

    return fp_log__, speeds_filtered

def complete_fixed_point_analysis(model, results, 
                                  EPS1 = 1e-7, EPS2 = 1e-1, EPS3 = 1.0, get_fixed_points_only=False, 
                                  partition='train', to_plot=True, 
                                  session_id=None  # needed for analysing mode importance
                                  ):
    
    
    if 'bias_sample' not in results[partition].keys():
        results = add_zero_bias(model, results)
        
    # pad zeros to bias corresponding to dimensions for time-varying input
    bias_padded = np.concatenate([results[partition]['bias_sample'], 
                  np.zeros((results[partition]['bias_sample'].shape[0], model.tv_input_dim))], axis=1)

    fixed_points, speed_fps = find_fixed_pts_contextual_lfads(model, 
                                                              bias_padded, 
                                                              results[partition]['state'],
                                                              noise_state=0, n_inits_per_example=20,
                                                              eps_list=[1e-7], max_steps=60000, 
                                                              lr=1) # lr=1)
    # See the speed distribution across fixed points
    if to_plot:
        plt.figure()
        plt.semilogy(np.sort(np.concatenate(speed_fps, axis=0)))
        plt.xlabel('fixed point')
        plt.ylabel('log (speed)')
        plt.show()

    # filter based on speeed 
    fp_log__, speeds_filtered = filter_fixedpoints(fixed_points, speed_fps, EPS1 = 1e-7, EPS2 = 1e-1, EPS3 = 1.0)

    # log to results
    results[partition].update({'fixed_points': fixed_points, 'fixed_points_filtered': fp_log__,
                               'fixed_point_speeds': speed_fps, 'fixed_points_filtered_speeds': speeds_filtered})

    if get_fixed_points_only:
        # only fixed point finding
        return results
    
    # Get jacobian around fixed points
    jacobians = find_jacobians_contextual_lfads(model, 
                                                results[partition]['fixed_points_filtered'], 
                                                bias_padded)

    results[partition].update({'jacobians': jacobians})


    # Get mode importance
    '''
    mode_mses = analyse_mode_importance(model, session_id,
                                        results[partition]['firing_rate'], 
                                        results[partition]['state'], 
                                        results[partition]['fixed_points_filtered'], 
                                        results[partition]['jacobians'])

    results[partition].update({'mode_mses': mode_mses})
    '''
    
    
    return results



def select_fixedpoints(r, d, fixed_point_key = 'fixed_points', fixed_point_speed_key = 'fixed_point_speeds', cues_key='cues'):
    
    # Compile cues from the dataset
    cues = np.array(d[cues_key])

    
    # Initialize lists to collect cues, trial IDs, and fixed points
    cues_ = []
    trial_ids_ = []
    fps = []
    speeds_ = []

    # Iterate over selected indices
    for i in range(len(r[fixed_point_key])):

        if len(r[fixed_point_speed_key][i]) == 0:
            continue

        # Get the slowest fixed point
        ifp = np.argmin(r[fixed_point_speed_key][i])

        # Append fixed points, expanded to include the first dimension
        fps += [np.expand_dims(r[fixed_point_key][i][ifp, ...], 0)]

        n_fps = 1  # Number of fixed points to consider per example
        # Append cue values repeated n_fps times
        cues_ += [cues[i]] * n_fps
        # Append trial IDs repeated n_fps times
        trial_ids_ += [d['trial_ids'][i] * np.ones(n_fps)]
        # Append speeds repeated n_fps times
        speeds_ += [r[fixed_point_speed_key][i][ifp] * np.ones(n_fps)]

    # Concatenate lists to form arrays
    fps = np.concatenate(fps, axis=0)
    # cues_ = np.concatenate(cues_, axis=0)
    trial_ids_ = np.concatenate(trial_ids_, axis=0)
    speeds_ = np.concatenate(speeds_, axis=0)

    return fps, cues_, trial_ids_, speeds_
