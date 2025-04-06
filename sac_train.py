# train SAC in CityLearn on all 2022 data 

# packages
import os
import time
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import joblib
from ipywidgets import IntProgress
from typing import List

# CityLearn
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.reward_function import RewardFunction
from citylearn.wrappers import NormalizedObservationWrapper
from citylearn.wrappers import StableBaselines3Wrapper

# baseline RL algorithms
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# helper functions
from kpi_utils import get_kpis, plot_building_kpis, plot_district_kpis
from kpi_utils import plot_building_load_profiles, plot_district_load_profiles, plot_battery_soc_profiles, plot_simulation_summary
from env_utils import set_n_buildings_2022
from obs_utils import get_n_observations
from rlc_utils import CustomReward

data_dir = '/home/jt9744/projects/thesis/CityLearn_thesis/data/datasets'
DATASET_NAME = 'citylearn_challenge_2022_phase_all'
ds = DataSet()
training_schema = ds.get_schema(DATASET_NAME)
training_schema['central_agent'] = True
train_env = CityLearnEnv(training_schema)
N_EPISODES = 5

# check initialization
print('Current time step:', train_env.time_step)
print('Number of episodes:', N_EPISODES)
print('Environment number of time steps:', train_env.time_steps)
print('Environment uses central agent:', train_env.central_agent)
print('Number of buildings:', len(train_env.buildings))
print('Electrical storage capacity:', {
    b.name: b.electrical_storage.capacity for b in train_env.buildings
})
print('Electrical storage nominal power:', {
    b.name: b.electrical_storage.nominal_power for b in train_env.buildings
})

# logging
log_dir = "./sb3_logs/sac" # Directory for Monitor logs and TensorBoard logs
os.makedirs(log_dir, exist_ok=True)

# initialize environment
train_env.reward_function = CustomReward(train_env)
train_env = NormalizedObservationWrapper(train_env) # ensure all observations that are served to the agent are min-max normalized between [0, 1] and cyclical observations e.g. hour, are encoded using the sine and cosine transformation.
train_env = StableBaselines3Wrapper(train_env) # ensures observations, actions and rewards are served in manner that is compatible with Stable Baselines3 interface
train_env = Monitor(train_env, filename=os.path.join(log_dir, "monitor.csv"))
observations, info = train_env.reset()

# initialize model 
sac_model = SAC(
    policy='MlpPolicy',
    env=train_env,
    device='cuda',
    seed=0,
    verbose=1, # Set to 1 to see SB3 logging output, including Monitor info
    tensorboard_log=os.path.join(log_dir, "logs/")
)

# train model
start_time = time.time()
sac_model = sac_model.learn(
    total_timesteps = training_schema['simulation_end_time_step'] * N_EPISODES,
    log_interval=1
)
print(f"Total time taken to train SAC for {training_schema['simulation_end_time_step'] * N_EPISODES} steps: {(time.time()-start_time) / 60:.2f} minutes")

# save model
model_save_path = os.path.join(log_dir, "sac_trained_model")
sac_model.save(model_save_path)
print(f"Model saved to {model_save_path}")