# Set local runtime configurations
import os
from absl import logging
import gin
from smart_control.utils import controller_reader
from smart_control.utils import histogram_reducer

histogram_parameters_tuples = (
        ('zone_air_temperature_sensor',(285., 286., 287., 288, 289., 290., 291., 292., 293., 294., 295., 296., 297., 298., 299., 300.,301,302,303)),
        ('supply_air_damper_percentage_command',(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
        ('supply_air_flowrate_setpoint',( 0., 0.05, .1, .2, .3, .4, .5,  .7,  .9)),
    )

def logging_info(*args):
    logging.info(*args)
    print(*args)

data_path = "/home/derek/sbsim/smart_control/configs/resources/sb1/" #@param {type:"string"}
metrics_path = "/home/derek/sbsim/PPO/PPO_Batch/metrics/" #@param {type:"string"}
output_data_path = '/home/derek/sbsim/PPO/PPO_Batch/output_data/' #@param {type:"string"}
root_dir = "/home/derek/sbsim/PPO/PPO_Batch/" #@param {type:"string"}

@gin.configurable
def get_histogram_reducer():

    reader = controller_reader.ProtoReader(data_path)

    hr = histogram_reducer.HistogramReducer(
        histogram_parameters_tuples=histogram_parameters_tuples,
        reader=reader,
        normalize_reduce=True,
        )
    return hr

def remap_filepath(filepath) -> str:
    return filepath

if __name__ == "__main__":
    pass