### run MPC implementation experiments (n_buildings = 2,4, 8)

# import packages
import os
import time
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet

# helper functions
from kpi_utils import get_kpis, plot_building_kpis, plot_district_kpis
from kpi_utils import plot_building_load_profiles, plot_district_load_profiles, plot_battery_soc_profiles, plot_simulation_summary
from env_utils import set_n_buildings_2022
from mpc_utils import mpc_district_optimization_2022

# packages for MPC
import cvxpy as cp
import joblib

# load data
data_dir = '/home/jt9744/projects/thesis/CityLearn_thesis/data/datasets'
DATASET_NAME = 'citylearn_challenge_2022_phase_all'
ds = DataSet()
schema = ds.get_schema(DATASET_NAME)

# run experiment
for n_buildings in [2, 4, 8]:
    # modify schema
    print(f'Running experiment for {n_buildings} buildings', flush=True)
    schema, buildings = set_n_buildings_2022(schema, n_buildings)
    print('Selected buildings:', buildings, flush=True)
    # use centralized control
    schema['central_agent'] = True

    # initialize env
    env = CityLearnEnv(schema)

    # check initialization
    print('Current time step:', env.time_step, flush=True)
    print('environment number of time steps:', env.time_steps, flush=True)
    print('environment uses central agent:', env.central_agent, flush=True)
    print('Number of buildings:', len(env.buildings), flush=True)

    # control loop 
    obs, info = env.reset() # standard gym reset
    rewards = []
    total_actions_log = []
    terminated, truncated = False, False
    building_datasets = {}
    start_time = time.time()
    
    for b in env.buildings:
        building_datasets[b.name] = pd.read_csv(f'{data_dir}/{DATASET_NAME}/{b.name}.csv')
    
    while not (terminated or truncated):
        current_time_step = env.time_step
        actions = mpc_district_optimization_2022(env.buildings, building_datasets, current_time_step, env.schema)
        if actions is None:
            print(f"Error: MPC optimization failed at step {current_time_step}. Stopping simulation.", flush=True)
            break
        obs, reward, terminated, truncated, info = env.step([actions])
        total_actions_log.append(actions)
        rewards.append(reward)
        if (current_time_step % (24*7*4) == 0 and current_time_step !=0) or terminated or truncated: # every one week or when done 
            elapsed_time = time.time() - start_time
            print(f"Step: {current_time_step}/{schema['simulation_end_time_step']}, Time elapsed: {(elapsed_time/60):.1f} minutes, Estimated time remaining: {((((schema['simulation_end_time_step']-current_time_step)/current_time_step)*elapsed_time)/60):.1f} minutes", flush=True)
    end_time = time.time()
    rewards = [r[0].item() if type(r[0]) == np.float32 else r[0] for r in rewards]

    total_sim_time = end_time - start_time
    print(f"\nSimulation finished.", flush=True)
    print(f"Total time steps executed: {env.time_step}", flush=True)
    print(f"Total time taken: {total_sim_time / 60:.2f} minutes", flush=True)
    print(f"Average time per step: {total_sim_time / (env.time_step):.3f} seconds", flush=True)
    print(f"Total accumulated reward: {sum(rewards):.2f}", flush=True)
    
    # Convert logged actions to numpy array
    total_actions_log_np = np.array(total_actions_log)

    # evaluate
    performance = env.evaluate()
    
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
    results_dir = f'results/results_mpc_2022_centralized/{n_buildings}_buildings'
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nSaving results to {results_dir}...", flush=True)
    joblib.dump(total_actions_log_np, os.path.join(results_dir, 'total_actions.joblib'))
    joblib.dump(rewards, os.path.join(results_dir, 'rewards.joblib'))
    performance.to_csv(os.path.join(results_dir, 'performance_kpis.csv'), index=False)
    with open(f'{results_dir}/kpi_metadata', "w") as f:
        json.dump(kpis, f, indent=4)
        
    # set all plotted figures without margins
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0

    # Plot actions performance
    plt.figure(figsize=(12, 6))
    # Transpose for plotting: actions on y-axis, time on x-axis
    plt.imshow(total_actions_log_np.T, aspect='auto', interpolation='none', cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Action Value (Battery)')
    plt.xlabel('Simulation Timestep')
    plt.ylabel('Building Action Index')
    plt.title(f'Centralized MPC Actions ({DATASET_NAME})')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'mpc_actions_heatmap.png'))

    # plot and save simulation summary
    plots = plot_simulation_summary({'env': env}, ret=True)
    for plot_name in plots:
        plots[plot_name].savefig(f'{results_dir}/{plot_name}')
    
    # plot and save electricity prices
    plt.plot(env.buildings[0].pricing.electricity_pricing)
    plt.ylim(0,1)
    plt.ylabel('Electricity Price ($)')
    plt.xlabel('Time step')
    plt.title('Electricity Price Profile')
    plt.savefig(os.path.join(results_dir, 'electricity_price.png'))

    # plot and save carbon intensity
    plt.plot(env.buildings[0].carbon_intensity.carbon_intensity)
    plt.ylim(0,0.5)
    plt.ylabel('Carbon Intensity (kgCO2/kWh)')
    plt.xlabel('Time step')
    plt.title('Carbon Intensity Profile')
    plt.savefig(os.path.join(results_dir, 'carbon_intensity.png'))
    print('----------------------------------------------------------------------------------------\n', flush=True)