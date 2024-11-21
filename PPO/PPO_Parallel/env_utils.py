# utils for importing environment
import gin
import os
import numpy as np
import pandas as pd
from typing import Sequence
from smart_control.utils import controller_reader
from smart_control.environment import environment
from smart_control.proto import smart_control_normalization_pb2
from smart_control.proto import smart_control_building_pb2
from smart_control.utils import bounded_action_normalizer
from local_runtime_utils import data_path, remap_filepath, metrics_path
from smart_control.utils import reader_lib

def load_environment(gin_config_file: str):
    """Returns an Environment from a config file."""
    # Global definition is required by Gin library to instantiate Environment.
    global environment  # pylint: disable=global-variable-not-assigned
    with gin.unlock_config():
        gin.parse_config_file(gin_config_file)
        return environment.Environment()  # pylint: disable=no-value-for-parameter


def get_latest_episode_reader(
    metrics_path: str,
) -> controller_reader.ProtoReader:
    episode_infos = controller_reader.get_episode_data(metrics_path).sort_index()
    selected_episode = episode_infos.index[-1]
    episode_path = os.path.join(metrics_path, selected_episode)
    reader = controller_reader.ProtoReader(episode_path)
    return reader

@gin.configurable
def get_histogram_path():
    return data_path  


@gin.configurable
def get_reset_temp_values():
    reset_temps_filepath = remap_filepath(
        os.path.join(data_path, "reset_temps.npy")
    )

    return np.load(reset_temps_filepath)


@gin.configurable
def get_zone_path():
    return remap_filepath(
        os.path.join(data_path, "double_resolution_zone_1_2.npy")
    )


@gin.configurable
def get_metrics_path():
    return os.path.join(metrics_path, "metrics")


@gin.configurable
def get_weather_path():
    return remap_filepath(
        os.path.join(
            data_path, "local_weather_moffett_field_20230701_20231122.csv"
        )
    )
    
# @gin.configurable
def to_timestamp(date_str: str) -> pd.Timestamp:
    """Utilty macro for gin config."""
    return pd.Timestamp(date_str)


# @gin.configurable
def local_time(time_str: str) -> pd.Timedelta:
    """Utilty macro for gin config."""
    return pd.Timedelta(time_str)


# @gin.configurable
def enumerate_zones(
    n_building_x: int, n_building_y: int
) -> Sequence[tuple[int, int]]:
    """Utilty macro for gin config."""
    zone_coordinates = []
    for x in range(n_building_x):
        for y in range(n_building_y):
            zone_coordinates.append((x, y))
    return zone_coordinates


# @gin.configurable
def set_observation_normalization_constants(
    field_id: str, sample_mean: float, sample_variance: float
) -> smart_control_normalization_pb2.ContinuousVariableInfo:
    return smart_control_normalization_pb2.ContinuousVariableInfo(
        id=field_id, sample_mean=sample_mean, sample_variance=sample_variance
    )


# @gin.configurable
def set_action_normalization_constants(
    min_native_value,
    max_native_value,
    min_normalized_value,
    max_normalized_value,
) -> bounded_action_normalizer.BoundedActionNormalizer:
    return bounded_action_normalizer.BoundedActionNormalizer(
        min_native_value,
        max_native_value,
        min_normalized_value,
        max_normalized_value,
    )


# @gin.configurable
def get_zones_from_config(
    configuration_path: str,
) -> Sequence[smart_control_building_pb2.ZoneInfo]:
  """Loads up the zones as a gin macro."""
  with gin.unlock_config():
    reader = reader_lib_google.RecordIoReader(input_dir=configuration_path)
    zone_infos = reader.read_zone_infos()
    return zone_infos


# @gin.configurable
def get_devices_from_config(
    configuration_path: str,
) -> Sequence[smart_control_building_pb2.DeviceInfo]:
  """Loads up HVAC devices as a gin macro."""
  with gin.unlock_config():
    reader = reader_lib_google.RecordIoReader(input_dir=configuration_path)
    device_infos = reader.read_device_infos()
    return device_infos


histogram_parameters_tuples = (
        ('zone_air_temperature_sensor',(285., 286., 287., 288, 289., 290., 291., 292., 293., 294., 295., 296., 297., 298., 299., 300.,301,302,303)),
        ('supply_air_damper_percentage_command',(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
        ('supply_air_flowrate_setpoint',( 0., 0.05, .1, .2, .3, .4, .5,  .7,  .9)),
    )

time_zone = 'US/Pacific'
collect_scenario_config = os.path.join(data_path, "sim_config.gin")
print(collect_scenario_config)
eval_scenario_config = os.path.join(data_path, "sim_config.gin")
print(eval_scenario_config)

if __name__ == "__main__":
    pass