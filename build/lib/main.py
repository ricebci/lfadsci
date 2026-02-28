import os

import hydra
import wandb
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

# import sys
# sys.path.append('/oak/stanford/groups/henderj/nishalps/code/lfads_ci/src/')

import lfadsci
from lfadsci.run_utils import *

@hydra.main(config_path='configs', config_name='config')
def app(config):
    print(OmegaConf.to_yaml(config))

    #set the visible device to the gpu specified in 'args' (otherwise tensorflow will steal all the GPUs)
    if 'gpuNumber' in config:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # print(f'Setting CUDA_VISIBLE_DEVICES to %d' % config["gpuNumber"])
        os.environ["CUDA_VISIBLE_DEVICES"] = config['gpuNumber']

    # if 'Slurm' in HydraConfig.get().launcher._target_:
    #     # TF train saver doesn't support file name with '[' or ']'. So we'll use relative path here.
    #     config.outputDir = './'

    print('Output dir %s' % config.outputDir)
    os.makedirs(config.outputDir, exist_ok=True)

    # if 'wandb' in config and config.wandb.enabled:
    #     run = wandb.init(**config.wandb.setup,
    #                      config=OmegaConf.to_container(config, resolve=True),
    #                      sync_tensorboard=True,
    #                      resume=True)
    train(config)

def train(config):
    print('config', config)
    data = get_data(config)
    train_model(data, config)

if __name__ == "__main__":
    app()
