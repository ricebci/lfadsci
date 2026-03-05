import numpy as np
import scipy as sp
import scipy.io as sio

# # mPCA analysis
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

# import shared_utils
# from shared_utils import *
import lfadsci.shared_utils
from lfadsci.shared_utils import *

# List of datasets: 
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.07.08/data4_1ms/', 
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.07.08/data3/', 
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.07.26/data3/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.09.20/data3/', 
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.09.20/data3_later_blcks/', 
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.10.06/data/',
                
        
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.10.18/data2/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.10.25/data/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.10.25/data_15ms/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.11.01/data/', 
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.11.10/data/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.11.15/data/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.11.24/data/', 
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.12.08/data/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.12.13/data/',
        
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.10.18/data/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.01.10/data2/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.01.12/data/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.01.31/data/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.02/data/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.05/data/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.07/data/',
                    
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.16/data/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.23/data/',
    #              '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.28/data/',
        
    #             '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.10.18/data_pv_analysis/',
    #             '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.03.02/data/',
    #             '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.03.09/data_fast/',
    #             '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.03.09/data_slow/'


def get_data(data_dirs, make_trials_same_length=True, 
             time_steps_before_movement=0, 
             time_steps_after_movement=0, 
             num_fingers_moving=None):

    # Tasks
    FOUR_FINGERS = 0
    SINGLE_FINGERS = 1
    TWO_FINGERS = 2
    THREE_FINGERS_ISOLATE = 3
    THUMB_POS_VEL = 4

    # Decoder types
    POS_DEC = 0
    VEL_DEC = 1

    # Data prep
    HISTORY = time_steps_before_movement
    FUTURE = time_steps_after_movement

    DOWNSAMPLE = 1


    ## Load data
    import os

    task_dict = {'/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.07.08/data4_1ms/': FOUR_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.07.08/data3/': FOUR_FINGERS, 
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.07.26/data3/': FOUR_FINGERS, 
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.09.20/data3/': FOUR_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.09.20/data3_later_blcks/': FOUR_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.10.06/data/': FOUR_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.10.18/data/': SINGLE_FINGERS,
                
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.10.18/data2/': TWO_FINGERS, 
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.10.25/data/': TWO_FINGERS, 
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.10.25/data_15ms/': TWO_FINGERS, 
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.11.01/data/': TWO_FINGERS, 
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.11.01/data_15ms/': TWO_FINGERS, 
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.11.10/data/': TWO_FINGERS, 
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.11.15/data/': TWO_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.11.24/data/': TWO_FINGERS, 
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.12.08/data/': TWO_FINGERS, 
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.12.13/data/': TWO_FINGERS, 
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.01.10/data/' : TWO_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.01.10/data_latest/': TWO_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.01.10/data2/' : SINGLE_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.01.10/data2_latest/': SINGLE_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.01.12/data/': SINGLE_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.01.12/data_latest/': SINGLE_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.01.31/data/': SINGLE_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.01.31/data_latest/': SINGLE_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.02/data/': SINGLE_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.02/data_latest/': SINGLE_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.05/data/': SINGLE_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.05/data_latest/': SINGLE_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.07/data/': SINGLE_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.07/data_latest/': SINGLE_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.16/data/': SINGLE_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.16/data_latest/': SINGLE_FINGERS,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.23/data/': THREE_FINGERS_ISOLATE,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.23/data_latest/': THREE_FINGERS_ISOLATE,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.28/data/': THREE_FINGERS_ISOLATE,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.02.28/data_latest/': THREE_FINGERS_ISOLATE,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.03.02/data/': THREE_FINGERS_ISOLATE,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.03.02/data_latest/': THREE_FINGERS_ISOLATE,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2021.10.18/data_pv_analysis/': THUMB_POS_VEL,
                
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.03.09/data_fast/': THREE_FINGERS_ISOLATE,
                '/oak/stanford/groups/henderj/nishalps/gesture/t5.2022.03.09/data_slow/': THREE_FINGERS_ISOLATE,
                }

    # if 'data_all' not in locals():
    data_all = {}

    for session_id, data_dir in enumerate(data_dirs):
        print(data_dir)
        files = os.listdir(data_dir)
        # sort files
        bl_num = [iff.split('-')[-1].split('_')[-1].split('.')[0][2:] for iff in files]
        print(bl_num)
        bl_num = [int(bl) for bl in bl_num]
        idx =np.argsort(bl_num)
        files = [files[iidx] for iidx in idx]
        print(files)
        
        X = []
        Y = []
        Z = []
        X_continuous = []
        Y_continuous = []
        Z_continuous = []
        click_enabled = []
        click_delivered = []
        start_tms = []
        trial_phase = []
        block_id = []

    #     if data_dir not in reload_sessions:
    #         print(f'Skipping {data_dir}')
    #         continue
            
            
        for iifile, ifile in enumerate(files):
            print(ifile)
            dat = sio.loadmat(os.path.join(data_dir, ifile))
            print(dat.keys())

            X_continuous += [dat['X_continuous']]
            Y_continuous += [dat['Y_continuous']]
            Z_continuous += [dat['target_continuous']]
            trial_phase += [dat['trial_phase']]
            n_trials = dat['X_batches'].shape[1]
            start_tms += [dat['start_times'][0, -n_trials:]]
            
            
            if not 'click_phase' in dat.keys():
                print('No Click found')
                time_pts = dat['X_continuous'].shape[0]
                click_enabled +=  [np.ones((time_pts, 1))] #[np.zeros((time_pts, 1))]
                click_delivered += [np.zeros((time_pts, 1))]
            else:
                print('Click found')
                if len(dat['click_phase'].shape) == 1:
                    dat['click_phase'] = np.expand_dims(dat['click_phase'], 1)
                click_enabled += [0 * dat['click_phase']  + 1] #np.max(dat['click_phase']) > 0]
                click_delivered += [dat['click_phase'] == 2]
            
        
        # zero-mean data
    #     Y_means = [y.mean(0) for y in Y_continuous]
    #     Y_continuous = [y - y.mean(0) for y in Y_continuous]
        
        if DOWNSAMPLE != 1:
            print('!!!!!!! Downsampling !!!!!!')
            Y_continuous = [rnn_utils.downsample_(y, DOWNSAMPLE, 'sum') for y in Y_continuous]
            X_continuous = [rnn_utils.downsample_(y, DOWNSAMPLE, 'avg') for y in X_continuous]
            Z_continuous = [rnn_utils.downsample_(y, DOWNSAMPLE, 'avg') for y in Z_continuous]
            click_enabled  = [rnn_utils.downsample_(y, DOWNSAMPLE, 'sample') for y in click_enabled]
            click_delivered  = [rnn_utils.downsample_(y, DOWNSAMPLE, 'sample') for y in click_delivered]
            
    #     # Fix thumb range
    #     print('!!!!!!! Fixing thumb range !!!!!!!!')
    #     for iz in range(len(Z_continuous)):
    #         Z_continuous[iz][:, 0] = (Z_continuous[iz][:, 0] - 0.5) * (0.5/0.7) + 0.5
        
        data_all.update({data_dir: {#'Y_means': Y_means,
                                    'X_continuous': X_continuous, 'Y_continuous': Y_continuous, 
                                    'Z_continuous': Z_continuous, 'trial_phase': trial_phase, 'start_tms': start_tms,
                                    'n_blocks': len(Z_continuous), 'session_id': session_id, 
                                    'click_enabled': click_enabled, 'click_delivered': click_delivered,
                                    'task_id': task_dict[data_dir], 
                                    'files': files, 
                                    'reloaded': True
                                }})



    # How many samples per phase, on average?
    phase_len = {0: [], 1: [], 2: [], 3: []}
    for isess, sess in enumerate(data_all.keys()):
        for blck_use in range(len(data_all[sess]['Z_continuous'])):
            phase  = int(np.squeeze(data_all[sess]['trial_phase'][blck_use][0]))
            tm = 1
            for itime in range(1, data_all[sess]['Z_continuous'][blck_use].shape[0]):
                if phase == np.squeeze(data_all[sess]['trial_phase'][blck_use][itime]):
                    tm = tm + 1
                else:
                    phase_len[phase] += [tm]
                    tm = 1
                    phase = int(np.squeeze(data_all[sess]['trial_phase'][blck_use][itime]))
                


    # remove very short samples - probably a mistake
    phase_len[1] = np.array(phase_len[1])
    phase_len[1] = phase_len[1][phase_len[1] > 2]

    phase_len[2] = np.array(phase_len[2])
    phase_len[2] = phase_len[2][phase_len[2] > 2]

    phase_len[3] = np.array(phase_len[3])
    phase_len[3] = phase_len[3][phase_len[3] > 2]

    prep_samples = int(np.median(phase_len[1]))
    move_samples = int(np.median(phase_len[2]))
    hold_samples = int(np.median(phase_len[3])) if len(phase_len[3]) > 0 else 0

    #HISTORY = int(np.median(phase_len[1]) + 10)   # PREP + 150ms before prep.
    # HISTORY = int(10)   # just 150ms before movement onset
    # HISTORY = np.int(prep_samples / 2)

    print('prep_samples', prep_samples, 
        'move_samples', move_samples, 
        'hold_samples', hold_samples, 
        'HISTORY', HISTORY)


    # convert into trials
    Z = []
    Y = []
    target = []
    start_pos = []
    move_time = []
    sess_id = []

    for isess, sess in enumerate(data_all.keys()):
        for blck_use in range(len(data_all[sess]['Z_continuous'])):

            in_trial = False
            moving = False
            block_started = False
            itrial = 0

            for itime in range(data_all[sess]['Z_continuous'][blck_use].shape[0]):

                if (not block_started) and np.abs(data_all[sess]['trial_phase'][blck_use][itime] - 1) < 0.01:
                    # print('STATE: Block started')
                    block_started = True
                
                    
                if block_started and (data_all[sess]['trial_phase'][blck_use][itime] == 2) and (not in_trial):
                    # print('STATE: Trial move started')
                    current_trial_move_start_time = itime
                    in_trial = True
                    moving = True

                if block_started and (data_all[sess]['trial_phase'][blck_use][itime] == 3) and (in_trial) and (moving):
                    # print('STATE: Trial move ended')
                    current_trial_move_end_time = itime
                    moving = False
                    in_trial = True

                if block_started and (data_all[sess]['trial_phase'][blck_use][itime] == 1) and (in_trial):
                    # print('STATE: Next trial prep started')
                    if moving:
                        current_trial_move_end_time = itime

                    current_trial_end_time = itime
                    in_trial = False
                    moving = False
                    Z += [data_all[sess]['Z_continuous'][blck_use][current_trial_move_start_time - HISTORY: 
                                                                current_trial_move_end_time + FUTURE, :]]
                    Y += [data_all[sess]['Y_continuous'][blck_use][current_trial_move_start_time - HISTORY: 
                                                                current_trial_move_end_time + FUTURE, ...]]
                    target += [data_all[sess]['Z_continuous'][blck_use][current_trial_end_time - 1, :]]
                    start_pos += [data_all[sess]['Z_continuous'][blck_use][current_trial_move_start_time - 1, :]]
                    # print(current_trial_end_time, blck_use,
                        # data_all[sess]['Z_continuous'][blck_use][current_trial_end_time - 1, :])
                    move_time += [current_trial_move_end_time - current_trial_move_start_time]
                    sess_id += [isess]


    task_prop = np.array(list(zip(target, move_time)))
    target = np.array(target)
    start_pos = np.array(start_pos)
    sess_id = np.array(sess_id)

    target = np.round(target * 10) / 10
    start_pos = np.round(start_pos * 10) / 10


    neural_trials = Y
    cued_trajectories = Z

    if make_trials_same_length:
        L = np.min(np.array([n.shape[0] for n in neural_trials]))
        L = int(L)
        neural_trials = [n[:L, :].astype(np.float32) for n in neural_trials]
        cued_trajectories = [n[:L, :].astype(np.float32) for n in cued_trajectories]

    cues_trials = []
    delays_trials = []
    for j in range(len(Y)):
        cue = ['%0.1f->%0.1f' % (start_pos[j][i], target[j][i]) for i in range(5)]
        cue = ','.join(cue)
        cues_trials += [cue]
        delays_trials += ['']
                
    delays_trials = cued_trajectories  # replace delays with cued trajectories 
    session_trials = np.ones((len(neural_trials), 1))

    if num_fingers_moving is not None:
        print('selecting trials with a subset of fingers moving.')
        trials_use = []
        num_dimensions_use = 2
        print(f'!!!!!!!measuring number of fingers moving using {num_dimensions_use} dimensions!!!!!!!!')
        for itrial in range(len(delays_trials)):
            n_fingers_moving_trial = ((np.abs(delays_trials[itrial][0, :num_dimensions_use] - 
                                              delays_trials[itrial][-1, :num_dimensions_use]) > 0).sum(-1))
            if n_fingers_moving_trial in num_fingers_moving:
                trials_use += [itrial]
    
        neural_trials_ = []
        cues_trials_ = []
        delays_trials_ = []
        session_trials_ = []
        for itrial in trials_use:
            neural_trials_ += [neural_trials[itrial]]
            cues_trials_ += [cues_trials[itrial]]
            delays_trials_ += [delays_trials[itrial]]
            session_trials_ += [session_trials[itrial]]

        neural_trials, cues_trials, delays_trials, session_trials = neural_trials_, cues_trials_, delays_trials_, session_trials_
        session_trials = np.array(session_trials)

    return neural_trials, cues_trials, delays_trials, session_trials

'''
# work in progress
def get_colors_thumb(cues):
    unique_cues = list(set(cues))
    [u.split(',')[0].split('->')[1]]

def get_colors_two_finger(cues):
    labels = {'0.5->0.0': -1, 
              '0.5->0.5': 0, 
              '0.5->1.0': 1}

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
                    
                    kinematics += [[labels[finger0], labels[finger1234]]]

    kinematics = np.array(kinematics)
    return kinematics

def analysis_single_pts_fingers(feature, cues, method='tdr', analysis='two_finger'):
    # get colors
    if analysis == 'thumb':
        kinematic_dict, cue_idx = get_colors_thumb(cues)
    elif analysis == 'two_finger':
        kinematic_dict, cue_idx = get_colors_two_finger(cues)
    else:
        raise ValueError('Invalid analysis.')


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

    plt.figure(figsize=(5, 5))
    for icue in np.unique(cue_idx_log):
        plt.plot(feature_2d[cue_idx_log == icue, 0].mean(0),
                    feature_2d[cue_idx_log == icue, 1].mean(0), '*',
                    color=cols_use[icue], markersize=20)

    for ipt in range(feature_2d.shape[0]):
        plt.plot(feature_2d[ipt, 0], feature_2d[ipt, 1], '.',
                    color=cols_use[cue_idx[cues[ipt]]], markersize=10, alpha=0.5)
    plt.title(col_title)

'''
