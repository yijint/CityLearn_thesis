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

# observations available in each building_obs:
# ['month',
#  'hour',
#  'day_type',
#  'indoor_dry_bulb_temperature',
#  'non_shiftable_load',
#  'dhw_demand',
#  'cooling_demand',
#  'heating_demand',
#  'solar_generation',
#  'daylight_savings_status',
#  'average_unmet_cooling_setpoint_difference',
#  'indoor_relative_humidity',
#  'occupant_count',
#  'indoor_dry_bulb_temperature_cooling_set_point',
#  'indoor_dry_bulb_temperature_heating_set_point',
#  'power_outage',
#  'comfort_band',
#  'indoor_dry_bulb_temperature_without_control',
#  'cooling_demand_without_control',
#  'heating_demand_without_control',
#  'dhw_demand_without_control',
#  'non_shiftable_load_without_control',
#  'indoor_relative_humidity_without_control',
#  'indoor_dry_bulb_temperature_cooling_set_point_without_control',
#  'indoor_dry_bulb_temperature_heating_set_point_without_control',
#  'hvac_mode',
#  'outdoor_dry_bulb_temperature',
#  'outdoor_relative_humidity',
#  'diffuse_solar_irradiance',
#  'direct_solar_irradiance',
#  'outdoor_dry_bulb_temperature_predicted_1',
#  'outdoor_dry_bulb_temperature_predicted_2',
#  'outdoor_dry_bulb_temperature_predicted_3',
#  'outdoor_relative_humidity_predicted_1',
#  'outdoor_relative_humidity_predicted_2',
#  'outdoor_relative_humidity_predicted_3',
#  'diffuse_solar_irradiance_predicted_1',
#  'diffuse_solar_irradiance_predicted_2',
#  'diffuse_solar_irradiance_predicted_3',
#  'direct_solar_irradiance_predicted_1',
#  'direct_solar_irradiance_predicted_2',
#  'direct_solar_irradiance_predicted_3',
#  'electricity_pricing',
#  'electricity_pricing_predicted_1',
#  'electricity_pricing_predicted_2',
#  'electricity_pricing_predicted_3',
#  'carbon_intensity',
#  'cooling_storage_soc',
#  'heating_storage_soc',
#  'dhw_storage_soc',
#  'electrical_storage_soc',
#  'net_electricity_consumption',
#  'cooling_electricity_consumption',
#  'heating_electricity_consumption',
#  'dhw_electricity_consumption',
#  'cooling_storage_electricity_consumption',
#  'heating_storage_electricity_consumption',
#  'dhw_storage_electricity_consumption',
#  'electrical_storage_electricity_consumption',
#  'cooling_device_efficiency',
#  'heating_device_efficiency',
#  'dhw_device_efficiency',
#  'indoor_dry_bulb_temperature_cooling_delta',
#  'indoor_dry_bulb_temperature_heating_delta']

# custom reward
class CustomReward(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        r"""Initialize CustomReward.

        Parameters
        ----------
        env: Mapping[str, CityLearnEnv]
            CityLearn environment instance.
        """

        super().__init__(env)

    def calculate(self, observations) -> List[float]:
        r"""Returns reward for most recent action.

        The reward is designed to minimize electricity cost.
        It is calculated for each building, i and summed to provide the agent
        with a reward that is representative of all n buildings.
        It encourages net-zero energy use by penalizing grid load satisfaction
        when there is energy in the battery as well as penalizing
        net export when the battery is not fully charged through the penalty
        term. There is neither penalty nor reward when the battery
        is fully charged during net export to the grid. Whereas, when the
        battery is charged to capacity and there is net import from the
        grid the penalty is maximized.

        Returns
        -------
        reward: float
            Reward for transition to current timestep.
        """

        reward_list = []

        for building_obs in observations:
            # use reward design and tuned hyperparams from https://arxiv.org/abs/2301.01148 
            consumption = building_obs['net_electricity_consumption']
            cost = consumption * building_obs['electricity_pricing']
            emissions = consumption * building_obs['carbon_intensity']
            battery_soc = building_obs['electrical_storage_soc']
            penalty = -(1.0 + np.sign(cost)*battery_soc) 
            e1, e2 = 1, 1
            w1, w2 = 1.0, 0.0
            reward = penalty*abs(w1*(cost**e1)+w2*(emissions**e2)) 
            reward_list.append(reward)
        return [sum(reward_list)]

# class CustomReward(RewardFunction):
#     def __init__(self, env: CityLearnEnv):
#         r"""Initialize CustomReward.

#         Parameters
#         ----------
#         env: Mapping[str, CityLearnEnv]
#             CityLearn environment instance.
#         """

#         super().__init__(env)

#     def calculate(self, observations) -> List[float]:
#         r"""Returns reward for most recent action.

#         The reward is designed to minimize electricity cost.
#         It is calculated for each building, i and summed to provide the agent
#         with a reward that is representative of all n buildings.
#         It encourages net-zero energy use by penalizing grid load satisfaction
#         when there is energy in the battery as well as penalizing
#         net export when the battery is not fully charged through the penalty
#         term. There is neither penalty nor reward when the battery
#         is fully charged during net export to the grid. Whereas, when the
#         battery is charged to capacity and there is net import from the
#         grid the penalty is maximized.

#         Returns
#         -------
#         reward: float
#             Reward for transition to current timestep.
#         """

#         reward_list = []

#         for building_obs in observations:
#             cost = building_obs['net_electricity_consumption']
#             battery_soc = building_obs['electrical_storage_soc']
#             penalty = -(1.0 + np.sign(cost)*battery_soc) 
#             reward = penalty*abs(cost) 
#             reward_list.append(reward)
#         return [sum(reward_list)]
