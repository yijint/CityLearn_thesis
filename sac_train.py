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
schema = ds.get_schema(DATASET_NAME)

n_buildings = 2 # 2, 4, or 8 
schema, buildings = set_n_buildings_2022(schema, n_buildings)
print('Number of buildings:', n_buildings)
print('Selected buildings:', buildings)
print(f'Number of observations: {get_n_observations(n_buildings)}')

schema['central_agent'] = True
train_env = CityLearnEnv(schema)
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
log_dir = f"./sb3_logs/sac/{n_buildings}_buildings" # Directory for Monitor logs and TensorBoard logs
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
    verbose=1,
    tensorboard_log=os.path.join(log_dir, "logs/"),
    # hyperparameters from https://arxiv.org/abs/2301.01148
    gamma=0.9,                  # Discount factor
    learning_rate=0.005,        # Learning rate
    ent_coef=0.2,               # Entropy coefficient (temperature)
    tau=0.05,                   # Target network update rate
    batch_size=256,             # Batch size
    buffer_size=100000,         # Replay buffer capacity
    policy_kwargs=dict(net_arch=[256, 256]) # 2 hidden layers of size 256 each
)

# train model
start_time = time.time()
sac_model = sac_model.learn(
    total_timesteps = schema['simulation_end_time_step'] * N_EPISODES,
    log_interval=1
)
print(f"Total time taken to train SAC for {schema['simulation_end_time_step'] * N_EPISODES} steps: {(time.time()-start_time) / 60:.2f} minutes")

# save model
model_save_path = os.path.join(log_dir, f"sac_model_{n_buildings}_buildings")
sac_model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# evaluate model 
eval_env = CityLearnEnv(schema)
eval_env = NormalizedObservationWrapper(eval_env)
eval_env = StableBaselines3Wrapper(eval_env)
observations, info = eval_env.reset()

sac_actions_list = []
rewards = []
terminated, truncated = False, False

start_time = time.time()
while not (terminated or truncated):
    actions, _ = sac_model.predict(observations, deterministic=True)
    observations, reward, terminated, truncated, info = eval_env.step(actions)
    sac_actions_list.append(actions)
    rewards.append(reward)
print(f"Total time taken for evaluation: {(time.time()-start_time) / 60:.2f} minutes")
rewards = [r.item() if type(r) == np.float32 else r for r in rewards]
sac_actions_list = np.stack(sac_actions_list)

performance = eval_env.unwrapped.evaluate()

# filter KPIs
kpis = {'all_time_peak_average': 'Average peak cost.',
        'carbon_emissions_total': 'Rolling sum of carbon emissions.', 
        'cost_total': 'Rolling sum of electricity monetary cost.',
        'daily_one_minus_load_factor_average': 'A measure of load variability / peakiness (daily average).', 
        'daily_peak_average': 'Average daily peak cost.',
        'electricity_consumption_total': 'Rolling sum of positive electricity consumption.',
        'monthly_one_minus_load_factor_average': 'A measure of load variability / peakiness (monthly average).',
        'ramping_average': 'Average rolling sum of absolute difference in net electric consumption between consecutive time steps',
        'zero_net_energy': 'Rolling sum of net electricity consumption'}
performance = performance[performance['cost_function'].isin(kpis.keys())].reset_index(drop=True)

# save evaluation
results_dir = f'results/results_sac_2022_centralized/{n_buildings}_buildings'
os.makedirs(results_dir, exist_ok=True)
print(f"\nSaving results to {results_dir}...")
joblib.dump(rewards, os.path.join(results_dir, 'rewards.joblib'))
joblib.dump(sac_actions_list, os.path.join(results_dir, 'total_actions.joblib'))
performance.to_csv(os.path.join(results_dir, 'performance_kpis.csv'), index=False)
with open(f'{results_dir}/kpi_metadata', "w") as f:
    json.dump(kpis, f, indent=4)

# set all plotted figures without margins
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

# Plot actions performance
plt.figure(figsize=(12, 6))
plt.imshow(sac_actions_list.T, aspect='auto', interpolation='none', cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Action Value (Battery)')
plt.xlabel('Simulation Timestep')
plt.ylabel('Building Action Index')
plt.title(f'Centralized SAC Actions ({DATASET_NAME})')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'sac_actions_heatmap.png'))

# Plot electricity pricing
plt.plot(eval_env.unwrapped.buildings[0].pricing.electricity_pricing)
plt.ylim(0,0.7)
plt.ylabel('Electricity Price ($)')
plt.xlabel('Time step')
plt.title('Electricity Price Profile')
plt.savefig(os.path.join(results_dir, 'electricity_price.png'))

# Plot carbon intensity
plt.plot(eval_env.unwrapped.buildings[0].carbon_intensity.carbon_intensity)
plt.ylim(0,0.7)
plt.ylabel('Carbon Intensity (kgCO2/kWh)')
plt.xlabel('Time step')
plt.title('Carbon Intensity Profile')
plt.savefig(os.path.join(results_dir, 'carbon_intensity.png'))

# Plot simulation summary
plots = plot_simulation_summary({'env': eval_env.unwrapped}, ret=True)
for plot_name in plots:
    plots[plot_name].savefig(f'{results_dir}/{plot_name}')