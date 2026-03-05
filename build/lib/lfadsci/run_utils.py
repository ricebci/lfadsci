   #### Import things.
import scipy as sp
import scipy.io as sio
import numpy as np

import matplotlib
from mpl_toolkits import mplot3d
# %matplotlib inline
import matplotlib.pyplot as plt
import scipy.ndimage

import scipy.stats

import pickle

# import sys
# sys.path.append('/oak/stanford/groups/henderj/nishalps/code/branch2/nptlbraingaterig/code/analysis/Nishal/Sequences/inferred_input')
# sys.path.append('/oak/stanford/groups/henderj/nishalps/code/branch2/nptlbraingaterig/code/analysis/Nishal/Utility/')
# # import decoders
# # from decoders import *
import importlib
# # importlib.reload(decoders)

# sys.path.append('/oak/stanford/groups/henderj/nishalps/code/branch2/nptlbraingaterig/code/analysis/Nishal/Gesture/2021_07_08/')

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# import rnn_utils 
# importlib.reload(rnn_utils)
# from rnn_utils import *

# import inferred_input_utils
# importlib.reload(inferred_input_utils)
# from inferred_input_utils import *

# import utils_multigen_no_input
# importlib.reload(utils_multigen_no_input)
# from utils_multigen_no_input import *

# import sys
# sys.path.append('/oak/stanford/groups/henderj/nishalps/code/lfads_ci/src/')

import lfadsci
import lfadsci.model 
from lfadsci.model import *

import lfadsci.shared_utils as shared_utils
from lfadsci.shared_utils import *

import lfadsci.utils_mcmaze as utils_mcmaze
import lfadsci.utils_fingers as utils_fingers
import lfadsci.utils_cvc as utils_cvc
import lfadsci.utils_ifg_operator as utils_ifg_operator
import lfadsci.utils_monkey_pfc as utils_monkey_pfc
import lfadsci.utils_pendulum as utils_pendulum

import datetime

from tqdm import tqdm
import colorsys

import tensorflow as tf
tf.__version__

def get_session(config, session_id=0):
    
    print('getting %s data' % (config['name']))

    if config['name'] == 'two_fingers':
        if 'num_fingers_moving' not in config.keys():
            num_fingers_moving = None
        else: 
            num_fingers_moving = config['num_fingers_moving']
        neural_trials, cues_trials, delays_trials, session_trials = utils_fingers.get_data([config['file']], 
                                                                             make_trials_same_length=True, 
                                                                             time_steps_before_movement=config['time_steps_before_movement'], 
                                                                             time_steps_after_movement=config['time_steps_after_movement'],
                                                                             num_fingers_moving=num_fingers_moving)
        
        session_trials = session_trials -1 + session_id
        return {'neural': neural_trials, 'cues': cues_trials, 
                'delays': delays_trials, 'session_id': session_trials, 
                'task': 'finger_t5'}

    if config['name'] == 'maze':
        spikes_trials, hand_pos_trials, hand_vel_trials, label_trials = utils_mcmaze.get_data(bin_width=config['bin_width'],
                                                                                   total_length=config['total_length'], 
                                                                                   pre_length=config['pre_length']) 
        session_trials =  np.array([[1] for _ in spikes_trials])
        session_trials = session_trials -1 + session_id
        return {'neural': spikes_trials, 'cues': label_trials, 
               'delays': hand_vel_trials, #[[''] for _ in spikes_trials], use velocity for delays
               'session_id': session_trials, 'task': 'maze_monkey'} 


    if config['name'] == 'pendulum':
        neural_trials, cues_trials, delays_trials, session_trials = utils_pendulum.get_data(n_trials_total=1000)
        session_trials = session_trials -1 + session_id
        return {'neural': neural_trials, 'cues': cues_trials, 
                'delays': delays_trials, 'session_id': session_trials, 
                'task': 'pendulum'}


    if config['name'] == 'monkey_pfc':
        neural_trials, cues_trials, delays_trials, session_trials = utils_monkey_pfc.get_data_monkey_pfc(filename=config['file'], 
                    T_start=config['T_start'], 
                    bin_size=config['bin_size'], 
                    remove_duplicate_channels=True)
        session_trials = session_trials -1 + session_id
        return {'neural': neural_trials, 'cues': cues_trials, 
                'delays': delays_trials, 'session_id': session_trials, 
                'task': 'monkey_pfc'}

    if config['name'] == 'cvc_t12_2023_06_29':
        neural_trials, cues_trials, delays_trials, session_trials = utils_cvc.get_data_t12_2023_06_29()
        session_trials = session_trials -1 + session_id
        return {'neural': neural_trials, 'cues': cues_trials, 
                'delays': delays_trials, 'session_id': session_trials, 
                'task': 'cvc_t12_2023_06_29'}

    if config['name'] == 'cvc_t15_2023_10_27':
        neural_trials, cues_trials, delays_trials, session_trials = utils_cvc.get_data_t15_2023_10_27(trial_stop_=200, channels=np.arange(256))
        session_trials = session_trials -1 + session_id
        return {'neural': neural_trials, 'cues': cues_trials, 
                'delays': delays_trials, 'session_id': session_trials, 
                'task': 'cvc_t15_2023_10_27'}

    if config['name'] == 'cvc_t15_2024_03_01':
        neural_trials, cues_trials, delays_trials, session_trials = utils_cvc.get_data_t15_2024_03_01(trial_stop_=200, channels=np.arange(256))
        session_trials = session_trials -1 + session_id
        return {'neural': neural_trials, 'cues': cues_trials, 
                'delays': delays_trials, 'session_id': session_trials, 
                'task': 'cvc_t15_2024_03_01'}    
    
    if config['name'] == 'ifg_verb':
        neural_trials, cues_trials, delays_trials, session_trials = utils_ifg_operator.get_data_ifg(
            channels=np.arange(128, 256), features=['binnedTX'], task='verb_conjugation')
        session_trials = session_trials -1 + session_id
        return {'neural': neural_trials, 'cues': cues_trials, 
                'delays': delays_trials, 'session_id': session_trials, 
                'task': 'ifg_verb'} 

    if config['name'] == '6v_verb':
        neural_trials, cues_trials, delays_trials, session_trials = utils_ifg_operator.get_data_ifg(
            channels=np.arange(128), features=['binnedTX'], task='verb_conjugation')
        session_trials = session_trials -1 + session_id
        return {'neural': neural_trials, 'cues': cues_trials, 
                'delays': delays_trials, 'session_id': session_trials, 
                'task': '6v_verb'} 

    if config['name'] == 'ifg_noun':
        neural_trials, cues_trials, delays_trials, session_trials = utils_ifg_operator.get_data_ifg(
            channels=np.arange(128, 256), features=['binnedTX'], task='noun_pluralization')
        session_trials = session_trials -1 + session_id
        return {'neural': neural_trials, 'cues': cues_trials, 
                'delays': delays_trials, 'session_id': session_trials, 
                'task': 'ifg_noun'} 

    if config['name'] == '6v_noun':
        neural_trials, cues_trials, delays_trials, session_trials = utils_ifg_operator.get_data_ifg(
            channels=np.arange(128), features=['binnedTX'], task='noun_pluralization')
        session_trials = session_trials -1 + session_id
        return {'neural': neural_trials, 'cues': cues_trials, 
                'delays': delays_trials, 'session_id': session_trials, 
                'task': '6v_noun'} 


def get_data(config):
    
    if config['dataset']['name'] == 'multiple_datasets':
        data_list = []
        data_weight = []
        data_val_weight = []
        for isess in range(len(config['dataset']['sessions'])):
            
            if 'session_id' in config['dataset']['sessions'][isess].keys():
                session_id_use = config['dataset']['sessions'][isess]['session_id']
            else:
                session_id_use = isess
            
            data_list += [get_session(config['dataset']['sessions'][isess], session_id=session_id_use)]
            
            if 'weight' in config['dataset']['sessions'][isess].keys():
                data_weight += [config['dataset']['sessions'][isess]['weight']]
            else:
                data_weight += [np.nan]

            if 'val_weight' in config['dataset']['sessions'][isess].keys():
                data_val_weight += [config['dataset']['sessions'][isess]['val_weight']]
            else:
                data_val_weight += [np.nan]

        if np.isnan(np.sum(data_weight)):
            data_weight = np.ones(len(data_list)).astype(np.float32)

        if np.isnan(np.sum(data_val_weight)):
            data_val_weight = np.ones(len(data_list)).astype(np.float32)
            
    else:
        data_list = [get_session(config['dataset'], session_id=0)]
        data_weight = [1]
        data_val_weight = [1]

    print('session_ids', [d['session_id'][0] for d in data_list]) # TODO: verify

    datagenerator_combined, datasets_list = shared_utils.combine_datasets(data_list, 
                                                            batch_sz=config['batch_size'], 
                                                            train_frac=config['train_frac'], 
                                                            val_frac=config['val_frac'], 
                                                            test_frac=config['test_frac'],
                                                            seed=config['data_seed'], 
                                                            data_weight=data_weight)

    return {'datagenerator': datagenerator_combined, 'datasets': datasets_list, 'data_val_weight': data_val_weight}

# def get_data_old(config):
#     #### Load data
#     print('Getting data .. ')
#     if config['dataset']['name'] == 'multiple_datasets':
#         from IPython import embed; embed()

#     elif config['dataset']['name'] == 'maze':
#         spikes_trials, hand_pos_trials, hand_vel_trials, _ = utils_mcmaze.get_data(bin_width=config['dataset']['bin_width'],
#                                                                                    total_length=config['dataset']['total_length'], 
#                                                                                    pre_length=config['dataset']['pre_length']) 
#         label_trials = hand_vel_trials    

#         #### Make data generator
#         datagenerators, datasets = shared_utils.get_data_generator2(spikes_trials, 
#                                                 [[0] for _ in spikes_trials], 
#                                                 label_trials, 
#                                                 [[''] for _ in spikes_trials], 
#                                                 batch_sz=config['batch_size'], # None 
#                                                 train_frac=config['train_frac'], 
#                                                 val_frac=config['val_frac'], 
#                                                 test_frac=config['test_frac'], 
#                                                 seed=config['seed'])

#         data = {'spikes_trials': spikes_trials, 
#                 'hand_pos_trials': hand_pos_trials, 
#                 'hand_vel_trials': hand_vel_trials,
#                 'label_trials': label_trials, 
#                 'datagenerators': datagenerators, 
#                 'partitions': datasets}

#     elif config['dataset']['name'] == 'two_fingers':
#         spikes_trials, cues_trials, delays_trials, session_trials = utils_fingers.get_data([config['dataset']['file']], 
#                                                                              make_trials_same_length=True, 
#                                                                              time_steps_before_movement=config['dataset']['time_steps_before_movement'], 
#                                                                              time_steps_after_movement=config['dataset']['time_steps_after_movement'])
        
#         #### Make data generator
#         datagenerators, datasets = get_data_generator2(spikes_trials, 
#                                                        [[0] for _ in spikes_trials],
#                                                         cues_trials, 
#                                                         delays_trials, 
#                                                         batch_sz=config['batch_size'], # None 
#                                                         train_frac=config['train_frac'], 
#                                                         val_frac=config['val_frac'], 
#                                                         test_frac=config['test_frac'], 
#                                                         seed=config['seed'])

#         data = {'spikes_trials': spikes_trials, 
#                 'cues_trials': cues_trials, 
#                 'delays_trials': delays_trials,
#                 'datagenerators': datagenerators, 
#                 'partitions': datasets}

#     elif config['dataset']['name'] == 'monkey_pfc':
#         neural_trials, cues_trials, delays_trials, session_trials = get_data_monkey_pfc(filename=config['dataset']['file'], 
#                     T_start=config['dataset']['T_start'], 
#                     bin_size=config['dataset']['bin_size'], 
#                     remove_duplicate_channels=True)

#         data_list = [{'neural': neural_trials, 
#                         'cues': cues_trials, 
#                         'delays': delays_trials, 
#                         'session_id': 0 * session_trials}]

#         print('starting to make data generators')
#         datagenerator_combined, datasets_list = combine_datasets(data_list, 
#                                                                 batch_sz=config['batch_size'], 
#                                                                 train_frac=config['train_frac'], 
#                                                                 val_frac=config['val_frac'], 
#                                                                 test_frac=config['test_frac'],
#                                                                 seed=config['seed'])
#         print('created data generators')

#         data = {'spikes_trials': neural_trials, 
#                 'cues_trials': cues_trials, 
#                 'delays_trials': delays_trials, 
#                 'datagenerators': datagenerator_combined, 
#                 'partitions': datasets_list[0]}
#     else:
#         raise ValueError('Not implemented')

#     print('Returning data')
#     return data                                                        



def train_model(data, config):
    print('Training .. ')
    
    model = load_model_from_config(config, 
                                   [dataset['train']['neural'].shape[-1] for dataset in data['datasets']], 
                                   model_filename=f'%s/model' % config['outputDir']) 

    # do fitting
    import time
    start_time = time.time()
    start_datetime = datetime.datetime.today()
    model_filename = f'%s/model' % config['outputDir']

    
    if config['mode'] == 'train':
        loss_np = train(data['datagenerator']['train'], model, 
                        data_test=[d['eval'] for d in data['datasets']],
                        lr_init=config['lr_init'], lr_stop=config['lr_stop'], lams=None, n_steps=config['n_steps'],
                        to_plot=False, 
                        kl_warmup_start=[config['model']['kl_warmup_start']], kl_warmup_end=[config['model']['kl_warmup_end']],
                        decay_factor=config['decay_factor'], gradient_clipping_norm=0.1, 
                        savefile=model_filename, 
                        n_eval_samples=None, 
                        patience_till_lr_decay=config['patience_till_lr_decay'], 
                        save_freq=config['save_freq'], 
                        data_val_weight=data['data_val_weight']
                        )
        
        print(f'Total training_time: {time.time() - start_time}')

        ##### Load checkpointed model
        model.load(model_filename)
        print('model loaded')

    ###### Posterior sample and average
    # Run model and get model parameters (i.c., bias, factors) using posterior sample and average. 
    # Note that state generator using estimated i.c. and bias.
    # results = compile_results(model, data['datasets'], n_samples=100, compute_kinematic_r2=False)
    results_list = []
    for dataset in data['datasets']:
        print('===================================')
        results = compile_results(model, dataset, n_samples=100, compute_kinematic_r2=False)
        results_list += [results]
    print('basic results compiled')

    pickle.dump({'results_list': results_list}, 
            open(f'%s/results_partial.pkl' % config['outputDir'] , 'wb'))

    ###### Fixed point finding analysis
    for ilist in range(len(results_list)):
        results = results_list[ilist]
        for partition in ['train', 'test', 'eval']:
            session_id = np.array(data['datasets'][ilist][partition]['session'][0, 0])
            print('Running fp finding for dataset: %d partition: %s' % (ilist, partition))
            results = complete_fixed_point_analysis(model, results, EPS1=1e-7, 
            get_fixed_points_only=False, partition=partition, to_plot=False, session_id=session_id)
        results_list[ilist] = results
    print('fixed point finding done')

    ####### Save results
    pickle.dump({'results_list': results_list}, 
                open(f'%s/results_full.pkl' % config['outputDir'] , 'wb'))
