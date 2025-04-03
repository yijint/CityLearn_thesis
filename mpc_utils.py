# helper functions for implemeting MPC with perfect forecast

# System operations
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

# packages for MPC
import cvxpy as cp
import joblib

def mpc_district_optimization_2022(buildings, building_datasets, time_step, schema):
    """
    Computes optimal charge/discharge actions for ALL buildings' batteries
    over an N-hour horizon to minimize total district cost and carbon emissions.
    Adapted for CityLearn 2022 (battery control only).

    Parameters:
    - buildings: List of CityLearn Building objects
    - building_datasets: Pre-loaded datasets for each building in buildings
    - time_step: Current simulation time step
    - schema: Environment schema

    Returns:
    - flat_actions: A flat list of battery actions [b1_batt_action, b2_batt_action, ...]
                   for the first timestep. Returns None if optimization fails.
    """
    # hyperparams based on https://doi.org/10.1051/e3sconf/202339604018
    N = 12 # Prediction horizon 
    COST_WEIGHT = 2.0 
    EMISSION_WEIGHT = 1.0 

    # initialize data structures
    num_buildings = len(buildings)
    all_constraints = []
    all_building_net_loads_vars = [] # Stores CVXPY expression for each building's net load
    all_battery_vars = {} # Dict: all_battery_vars[b_idx] stores battery variables
    ordered_action_vars = [] # Store action vars in the order expected by env.step

    # Determine Effective Horizon
    effective_N = min(N, schema['simulation_end_time_step'] - time_step)
    assert effective_N >= 0, 'effective_N cannot be negative'

    # 1. Gather Forecasts and Define Variables/Constraints for ALL buildings
    building_forecasts = []
    active_building_indices_map = {} # Map original building index to filtered index

    for b_idx, b in enumerate(buildings):
        # Map original index to the index in filtered lists (forecasts, net_loads)
        active_idx = len(building_forecasts)
        active_building_indices_map[b_idx] = active_idx

        ds = building_datasets[b.name]
        elec_load = ds['non_shiftable_load'].values[time_step:time_step + effective_N]
        pv_gen = ds['solar_generation'].values[time_step:time_step + effective_N]
        price = b.pricing.electricity_pricing[time_step:time_step + effective_N]
        carbon = b.carbon_intensity.carbon_intensity[time_step:time_step + effective_N]
        
        building_forecasts.append({
            'elec': elec_load, 'pv': pv_gen, 'price': price, 'carbon': carbon
        })

        # Battery Device Properties
        dev = b.electrical_storage
        capacity = dev.capacity
        efficiency = dev.efficiency
        soc_init = dev.soc[time_step] # Current state of charge in kWh        
        nominal_power = dev.nominal_power
        # Get min/max SOC limits
        max_discharge = dev.depth_of_discharge if hasattr(dev, 'depth_of_discharge') else 1.0 # Assumes DoD maps to min SoC limit
        if max_discharge:
            min_soc_factor = 1-max_discharge
        else:
            min_soc_factor = 0.0
        min_soc = min_soc_factor * capacity
        max_soc = capacity

        # CVXPY Variables for Battery
        action = cp.Variable(effective_N, name=f"action_b{b_idx}") # Action: [-1, 1] proportion of kWh
        soc = cp.Variable(effective_N + 1, name=f"soc_b{b_idx}")
        charge_energy = cp.Variable(effective_N, nonneg=True, name=f"charge_b{b_idx}")
        discharge_energy = cp.Variable(effective_N, nonneg=True, name=f"discharge_b{b_idx}")

        # Store action var for final extraction in the correct order
        ordered_action_vars.append(action)

        # Store all vars for objective/constraint calculation
        all_battery_vars[b_idx] = {
            'action': action, 'soc': soc, 'charge': charge_energy,
            'discharge': discharge_energy, 'capacity': capacity,
            'efficiency': efficiency, 'nominal_power': nominal_power
        }

        # CVXPY Constraints for Battery
        all_constraints.append(soc[0] == soc_init)
        all_constraints.append(cp.abs(action) <= 1) # Action interpretation [-1, 1]

        for t in range(effective_N):
            # The *target* net energy transfer (before efficiency) is action * capacity.
            # The actual charge/discharge energy variables must equal this target,
            # while also respecting power limits.

            target_net_energy_transfer = action[t] * capacity # kWh target
            all_constraints.append(charge_energy[t] - discharge_energy[t] == target_net_energy_transfer)

            # Apply Power Limits
            # The actual energy charged or discharged in 1 hour cannot exceed nominal power (kW * 1h = kWh)
            all_constraints.append(charge_energy[t] <= nominal_power)
            all_constraints.append(discharge_energy[t] <= nominal_power)

            # SoC Update (Physics)
            # Uses the actual charge/discharge energy achieved, considering efficiency.
            soc_update = efficiency * charge_energy[t] - discharge_energy[t] / efficiency
            all_constraints.append(soc[t + 1] == soc[t] + soc_update)

            # SoC Bounds
            all_constraints.append(soc[t + 1] >= min_soc)
            all_constraints.append(soc[t + 1] <= max_soc)
    
        # Calculate CVXPY expression for this building's net load
        forecast = building_forecasts[active_idx] # Use the mapped active index
        battery_vars = all_battery_vars[b_idx]

        # Net Load = Base Load - PV Generation + Battery Net Load 
        building_net_load = forecast['elec'] - forecast['pv'] + charge_energy - discharge_energy
        all_building_net_loads_vars.append(building_net_load) # Append in order of active buildings

    # 2. Define District Objective
    total_dollar_cost = 0
    total_emission_cost = 0

    # Iterate through the filtered lists using active index 'i'
    for i in range(len(all_building_net_loads_vars)):
        forecast = building_forecasts[i]
        building_net_load = all_building_net_loads_vars[i]

        # Cost: sum over time of positive net load * price
        building_cost = cp.sum(cp.multiply(cp.pos(building_net_load), forecast['price']))
        total_dollar_cost += building_cost

        # Emissions: sum over time of positive net load * carbon intensity
        building_emissions = cp.sum(cp.multiply(cp.pos(building_net_load), forecast['carbon']))
        total_emission_cost += building_emissions

    objective = cp.Minimize(COST_WEIGHT * total_dollar_cost + EMISSION_WEIGHT * total_emission_cost)

    # 3. Solve the Optimization Problem
    try:
        problem = cp.Problem(objective, all_constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
    except cp.error.SolverError as e:
        print(f"CVXPY SolverError at time step {time_step}: {e}")
    except Exception as e:
        print(f"Unexpected error during optimization at step {time_step}: {e}")

    # 4. Extract Actions for the First Timestep
    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        first_step_actions = []
        for action_var in ordered_action_vars: # Use the ordered list of action variables
             if action_var.value is not None:
                 action_val = np.clip(action_var.value[0], -1.0, 1.0) # Clip for safety
                 first_step_actions.append(action_val)
             else:
                 print(f"Warning: Optimal status but None value for action variable {action_var.name()} at step {time_step}. Using 0.0.")
                 first_step_actions.append(0.0)
        return first_step_actions
    else:
        print(f"Warning: Optimization problem status at step {time_step}: {problem.status}. Using 0.0 actions.")
        zero_actions = [0.0] * len(ordered_action_vars)
        return zero_actions