import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # -1 for cpu, currently available gpu 0, 1

import gin
from absl import app
import functools
os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'true'
import sys
# Add the directory for smart_control to sys.path
grandparent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(grandparent_dir)
import tensorflow as tf

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.agents.ppo import ppo_actor_network
from tf_agents.networks import value_network
from tf_agents.environments import ActionClipWrapper

from tf_agents.metrics import py_metrics
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import ppo_learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
import reverb
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

from smart_control.environment import environment
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_normalization_pb2
from smart_control.reward import electricity_energy_cost
from smart_control.reward import natural_gas_energy_cost
from smart_control.reward import setpoint_energy_carbon_regret
from smart_control.reward import setpoint_energy_carbon_reward
from smart_control.simulator import randomized_arrival_departure_occupancy
from smart_control.simulator import rejection_simulator_building
from smart_control.simulator import simulator_building
from smart_control.simulator import step_function_occupancy
from smart_control.simulator import stochastic_convection_simulator
from smart_control.utils import bounded_action_normalizer
from smart_control.utils import building_renderer
from smart_control.utils import controller_reader
from smart_control.utils import controller_writer
from smart_control.utils import conversion_utils
from smart_control.utils import observation_normalizer
from smart_control.utils import reader_lib
from smart_control.utils import writer_lib
from smart_control.utils import histogram_reducer
from smart_control.utils import environment_utils

from plotting_utils import *
from local_runtime_utils import *
from observer_classes import *
from env_utils import *

# Path to the saved greedy policy
policy_dir = "/home/derek/sbsim/PPO/PPO_Debug/policies/greedy_policy"

# Load the trained greedy policy
trained_policy = tf.saved_model.load(policy_dir)

eval_env = load_environment(eval_scenario_config)
# eval_env._metrics_path = metrics_path
eval_env._occupancy_normalization_constant = 125.0
eval_env = ActionClipWrapper(eval_env)

sequence_length = eval_env.steps_per_episode

eval_print_status_observer = PrintStatusObserver(
    status_interval_steps=1, environment=eval_env, replay_buffer=None
)

eval_actor = actor.Actor(
    eval_env,
    trained_policy,
    train_step=tf.Variable(0),
    steps_per_run=sequence_length,
    metrics=actor.eval_metrics(1),
    summary_dir=os.path.join(root_dir, 'eval'),
    summary_interval=1,
)

eval_actor.run_and_log()