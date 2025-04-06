"""
HANDLING OBSERVATIONS
---------------------
There are 24 district-level observations and 4 building-level observations, resulting in `24+4*n_buildings` observations. The first twenty observations are the first twenty observations in `district_obs` below, followed by the four observations (in `building_obs` below) for the first building, the last four observations in `district_obs`, and then the four observations (in `building_obs`) for the remaining buildings. For instance, for two buildings, the observation order would be: 

- 'month'
- ...
- 'carbon_intensity'
- 'non_shiftable_load' for the **first** building
- 'solar_generation' for the **first** building
- 'electrical_storage_soc' for the **first** building
- 'net_electricity_consumption' for the **first** building
- 'electricity_pricing'
- 'electricity_pricing_predicted_1'
- 'electricity_pricing_predicted_2'
- 'electricity_pricing_predicted_3'
- 'non_shiftable_load' for the **second** building
- 'solar_generation' for the **second** building
- 'electrical_storage_soc' for the **second** building
- 'net_electricity_consumption' for the **second** building
  
And, for four buildings, the observation order would be:

- 'month'
- ...
- 'carbon_intensity'
- 'non_shiftable_load' for the **first** building
- 'solar_generation' for the **first** building
- 'electrical_storage_soc' for the **first** building
- 'net_electricity_consumption' for the **first** building
- 'electricity_pricing'
- 'electricity_pricing_predicted_1'
- 'electricity_pricing_predicted_2'
- 'electricity_pricing_predicted_3'
- 'non_shiftable_load' for the **second** building
- 'solar_generation' for the **second** building
- 'electrical_storage_soc' for the **second** building
- 'net_electricity_consumption' for the **second** building
- 'non_shiftable_load' for the **third** building
- 'solar_generation' for the **third** building
- 'electrical_storage_soc' for the **third** building
- 'net_electricity_consumption' for the **third** building
- 'non_shiftable_load' for the **fourth** building
- 'solar_generation' for the **fourth** building
- 'electrical_storage_soc' for the **fourth** building
- 'net_electricity_consumption' for the **fourth** building
"""

# establish district and building observations 
district_obs_names_1 = ['month',
 'day_type',
 'hour',
 'outdoor_dry_bulb_temperature',
 'outdoor_dry_bulb_temperature_predicted_1',
 'outdoor_dry_bulb_temperature_predicted_2',
 'outdoor_dry_bulb_temperature_predicted_3',
 'outdoor_relative_humidity',
 'outdoor_relative_humidity_predicted_1',
 'outdoor_relative_humidity_predicted_2',
 'outdoor_relative_humidity_predicted_3',
 'diffuse_solar_irradiance',
 'diffuse_solar_irradiance_predicted_1',
 'diffuse_solar_irradiance_predicted_2',
 'diffuse_solar_irradiance_predicted_3',
 'direct_solar_irradiance',
 'direct_solar_irradiance_predicted_1',
 'direct_solar_irradiance_predicted_2',
 'direct_solar_irradiance_predicted_3',
 'carbon_intensity']

district_obs_names_2 = ['electricity_pricing',
 'electricity_pricing_predicted_1',
 'electricity_pricing_predicted_2',
 'electricity_pricing_predicted_3']

building_obs_names = ['non_shiftable_load',
 'solar_generation',
 'electrical_storage_soc',
 'net_electricity_consumption',]

# return number observations
def get_n_observations(n_buildings):
    return len(district_obs_names_1) +len(district_obs_names_2)+ len(building_obs_names)*n_buildings  # total number of observations
    
# return observations with metadata, given a list of observations and building NAMES
def extract_observation_metadata(observations_list, buildings):
    district_obs = {}
    building_obs = {}

    if len(observations_list)==1:
        print(f'Assuming that observations is in a list')
        observations = observations_list[0]
    else:
        observations = observations_list

    # extract first 20 district data
    for i, d_o in enumerate(district_obs_names_1):
        district_obs[d_o] = observations[i] #.item()

    # extract data for first building 
    building_obs[buildings[0]] = {}
    for i, b_o in enumerate(building_obs_names):
        building_obs[buildings[0]][b_o] = observations[len(district_obs_names_1)+i] #.item()

    # extract final 4 district data
    for i, d_o in enumerate(district_obs_names_2):
        district_obs[d_o] = observations[len(district_obs_names_1)+len(building_obs_names)+i] #.item()

    # extract data for remaining buildings 
    for i, b in enumerate(buildings):
        if i == 0:
            continue 
        building_obs[b] = {}
        for j, b_o in enumerate(building_obs_names):
            building_obs[b][b_o] = observations[len(district_obs_names_1)+(i)*len(building_obs_names)+len(district_obs_names_2)+j] #.item()

    return district_obs, building_obs