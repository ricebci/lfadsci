def interpolate_bias_ic_and_linearize2(model, inp_refs, ic_refs, trial_len, n_steps=10,
                           batch_size=10, noise_state_init=0, fp_lr=0.1, fp_maxsteps=60000,
                           eps_stop=1e-9, fp_speed_threshold=1e-9):
    

    steps = np.linspace(0, 1, n_steps)
    input_center = np.array(inp_refs).mean(0)
    ic_center = np.array(ic_refs).mean(0)
    
    states_init_log = []
    input_log = []
    ic_log = []
    step_log = []
    sample_log = []
    
    for isample in range(len(inp_refs)):

        for step in steps:
            
            for ibatch in range(batch_size):
                # interpolate
                input_ = inp_refs[isample] * step + input_center * (1 - step)
                ic_ = ic_refs[isample] * step + ic_center * (1 - step)

                input_log += [input_]
                ic_log += [ic_]
                step_log += [step]
                sample_log += [isample]
    
    input_log = np.array(input_log)
    ic_log = np.array(ic_log)
    step_log = np.array(step_log)
    sample_log = np.array(sample_log)
    
    print('input', input_log.shape, 'ic', ic_log.shape, 
          'step_log', step_log.shape, 
          'sample_log', sample_log.shape)
    
    # generate initial state for fixed point finding
    inp_time = np.zeros((1, trial_len, 1)) + np.expand_dims(input_log, 1)
    state_traj = model.generators[0](tf.convert_to_tensor(inp_time), 
                                     initial_state=model.ic_to_state[0](ic_log))
    state_traj = state_traj.numpy()
    states_init = []
    for ii in range(state_traj.shape[0]):
        t_choice = np.random.choice(trial_len, 1)[0]
        states_init += [state_traj[ii, t_choice, :] + 
                        np.random.randn(state_traj.shape[-1]) * noise_state_init]
    states_init = np.array(states_init)
    print('state_traj', state_traj.shape, 'states_init', states_init.shape)
    print('biases and initial states calculated')
    
    # estimate fixed points
    fp_, loss_sample_ = find_fixed_point(model.generators[0], 
                                         states_init, #
                                         input_log, 
                                         eps_stop=eps_stop, lr=np.float32(fp_lr), #lr=0.1 
                                         decay=1, 
                                         decay_steps=np.inf,
                                         max_steps=fp_maxsteps)
    
    fp_ = fp_[loss_sample_ < fp_speed_threshold]
    input_log = input_log[loss_sample_ < fp_speed_threshold]
    step_log = step_log[loss_sample_ < fp_speed_threshold]
    sample_log = sample_log[loss_sample_ < fp_speed_threshold]
    print('fixed point finding done')
    
    
    # Get jacobian around fixed points
    fp__ = [np.expand_dims(fp_[i, ...], 0) for i in range(fp_.shape[0])]
    jacobians = find_jacobians_contextual_lfads(model, 
                                                fp__, 
                                                input_log)
    print('jacobians calculated')
    
    return fp_, jacobians, step_log, input_log, sample_log

from scipy.optimize import linear_sum_assignment

def match_lams(lam1, lam2, label_lam1):
    # Calculate the distance matrix
    distance_matrix = np.abs(lam1[:, np.newaxis] - lam2)

    # Solve the linear sum assignment problem
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Create the matches with labels
    matches = [(label_lam1[i], col_ind[i]) for i in row_ind]
    
    # Get new labels
    order = np.argsort([m[1] for m in matches])
    labels = [matches[iorder][0] for iorder in order]
    
    return labels


def order_fixed_points(fp_, center_idx):
    # Calculate pairwise squared Euclidean distances between points in fp_
    distances = ((fp_[:, np.newaxis, :] - fp_[np.newaxis, :, :])**2).sum(-1)

    # Initialize the ordered indices starting with the center index
    ordered_idx = [(center_idx, None)]
    # Create a list of all indices and remove the center index
    remaining_idx = list(np.arange(len(fp_)))
    remaining_idx.remove(center_idx)

    # Iterate until all points are ordered
    while remaining_idx:
        # Extract indices of already ordered points
        added = [o[0] for o in ordered_idx]
        # Create a window of distances including only ordered points
        distances_window = distances[:, added]
        distances_window = distances_window[remaining_idx, :]
        
        # Find the minimum distance in the window
        dist_min = np.min(np.ndarray.flatten(distances_window))
        
        # Get indices of the remaining and ordered points with the minimum distance
        iremain, iorder = np.where(distances_window == dist_min)
        iremain = iremain[0]
        iorder = iorder[0]
        
        # Add the closest point and its reference to the ordered list
        ordered_idx += [(remaining_idx[iremain], added[iorder], dist_min)]
        
        # Remove the newly added point from the list of remaining indices
        remaining_idx.pop(iremain)
        # Uncomment the following line to print the process details
        # print('adding', ordered_idx[-1], 'remaining', len(remaining_idx))
        
    print('ordered fixed points')
    return ordered_idx


def match_modes_across_fixed_points(order_idx, jacobians):
    # Initialize a list to store mode indices for each fixed point
    mode_idx = [[]] * len(jacobians)
    
    # Match modes for the first element
    first_fp = order_idx[0][0]
    print(first_fp, len(jacobians))
    print(jacobians[first_fp])
    labels = list(np.arange(len(jacobians[first_fp][0]['lam'])))
    mode_idx[first_fp] = labels
    
    # Match modes for the second element onwards
    for ifp in range(1, len(order_idx)):
        new_fp, reference_fp = order_idx[ifp][0], order_idx[ifp][1]
        # Use the match_lams function to match modes of the new fixed point with the reference
        labels_new = match_lams(jacobians[new_fp][0]['lam'], 
                                jacobians[reference_fp][0]['lam'], 
                                mode_idx[reference_fp])
        # Update mode indices for the new fixed point
        mode_idx[new_fp] = labels_new #[l[1] for l in labels_new]
    
    
    print('modes matched')
    return mode_idx


def plot_modes(order_idx, mode_idx, jacobians):
    
    for ifp in range(1, len(order_idx)):
        new_fp, reference_fp = order_idx[ifp][0], order_idx[ifp][1]
        labels_new = mode_idx[new_fp]
        lam1 = jacobians[reference_fp][0]['lam']
        lam2 = jacobians[new_fp][0]['lam']

#         plt.plot(np.abs(jacobians[new_fp][0]['lam']), np.angle(jacobians[new_fp][0]['lam']), 'k.')
        for ilabel in mode_idx[reference_fp]:

            xpt = np.where(mode_idx[reference_fp] == ilabel)[0]
            ypt = np.where(labels_new == ilabel)[0]
            plt.plot([np.abs(lam1[xpt]), np.abs(lam2[ypt])], 
                     [np.angle(lam1[xpt]), np.angle(lam2[ypt])], 'k')
            
#             if ilabel == 77:
            plt.text(np.abs(lam2[xpt]), 
                     np.angle(lam2[xpt]), ilabel)


    
    plt.xlim([0.5, 1.1])
    plt.ylim([-0.5, 0.5])



#
fp_, jacobians, step_log, input_log, sample_log = interpolate_bias_ic_and_linearize2(model, 
                                results['train']['input'][:100, ...],
                                results['train']['ic'][:100, ...], 
                                trial_len=results['train']['state'].shape[1], 
                                n_steps=20, batch_size=1, 
                                noise_state_init=0, fp_lr=0.1, fp_maxsteps=60000,
                                eps_stop=1e-9, fp_speed_threshold=1e-8)

#
def interpolate_bias_fp_and_getJ(model, inputs, fps, n_steps=10):
    # inputs: samples x input_dimension
    # fps: fps x state_dimension
    
    
    steps = np.linspace(0, 1, n_steps)
    input_mean = np.array(inputs).mean(0)
    fps_mean = np.array(fps).mean(0)
    
    mode_idx_log = []
    for isample in range(len(inputs)):
        
        # interpolate
        fps_interpolate_log = []
        input_interpolate_log = []
        for step in steps:
            input_interpolate_log += [inputs[isample] * step + input_mean * (1 - step)]
            fps_interpolate_log += [ fps[isample] * step + fps_mean * (1 - step)]
        
        fps_interpolate_log = [np.expand_dims(f, 0) 
                               for f in fps_interpolate_log]
        input_interpolate_log = np.array(input_interpolate_log)
        
        print([f.shape for f in fps_interpolate_log])
        
        # find jacobian for each step in interpolation
        jacobians_interpolate_log = find_jacobians_contextual_lfads(model, 
                                                                    fps_interpolate_log, 
                                                                    input_interpolate_log)
        
        # Match modes
        n_jacs = len(jacobians_interpolate_log)
        order_idx = list(zip(range(1, n_jacs), range(n_jacs - 1)))
#         print(order_idx)
#         return [[j] for j in jacobians_interpolate_log]
    
        mode_idx = match_modes_across_fixed_points(order_idx, 
                                                   jacobians_interpolate_log)
                     
        # Note the modes for the last point
        mode_idx_log += [mode_idx[-1]] 
        
    return mode_idx_log



idx_select = [i for i in range(len(results['train']['fixed_points_filtered'])) 
                          if len(results['train']['fixed_points_filtered'][i]) > 0]
idx_select = np.array(idx_select)

mode_idx_log = interpolate_bias_fp_and_getJ(model, 
                                            results['train']['input'][idx_select, :], 
                                            [results['train']['fixed_points_filtered'][i][0] for i in idx_select], 
                                            n_steps=10)



center_idx = np.where(step_log == 0)[0][0]
ordered_idx = order_fixed_points(fp_, center_idx)
mode_idx = match_modes_across_fixed_points(ordered_idx, jacobians)


idx_use = [idx for idx in range(len(results[partition_use]['fixed_points_filtered'])) if 
                         len(results[partition_use]['fixed_points_filtered'][idx]) > 0]

fp_ = [results[partition_use]['fixed_points_filtered'][i][0] for i in idx_use]
fp_ = np.array(fp_)

jacs = [results[partition_use]['jacobians'][i] for i in idx_use]

center_idx = 0
ordered_idx = order_fixed_points(fp_, center_idx)
mode_idx = match_modes_across_fixed_points(ordered_idx, jacs)