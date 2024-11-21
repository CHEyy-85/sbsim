# Python Script for PPO to better utilize parallel env, which requires Python multiprocessing

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @title Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # -1 for cpu, currently available gpu 0, 1

import gin
from absl import app
import functools
os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'true'
import tensorflow as tf
grandparent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../"))
import sys
sys.path.append(grandparent_dir)
import tensorflow as tf

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.agents.ppo import ppo_actor_network
from tf_agents.environments import ParallelPyEnvironment
from tf_agents.environments import BatchedPyEnvironment
from tf_agents.networks import value_network

from tf_agents.metrics import py_metrics
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import table
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import ppo_learner
from tf_agents.utils import common
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.system import system_multiprocessing as multiprocessing

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



class CustomTFUniformReplayBuffer(tf_uniform_replay_buffer.TFUniformReplayBuffer):
    def __init__(
        self,
        data_spec,
        batch_size,
        max_length=1000,
        agent=None,  # Pass your agent here
        scope='CustomTFUniformReplayBuffer',
        device='cpu:*',
        table_fn=table.Table,
        dataset_drop_remainder=False,
        dataset_window_shift=None,
        stateful_dataset=False,
    ):
        """
        CustomTFUniformReplayBuffer that preprocesses dataset before return.
        Allow calculation of advantages and values to be done with correct temporal order before being shuffled in PPO_Learner.
        Provide compatability between vectorized environments and PPO_Learner work flow.
        """
        super(CustomTFUniformReplayBuffer, self).__init__(
            data_spec=data_spec,
            batch_size=batch_size,
            max_length=max_length,
            scope=scope,
            device=device,
            table_fn=table_fn,
            dataset_drop_remainder=dataset_drop_remainder,
            dataset_window_shift=dataset_window_shift,
            stateful_dataset=stateful_dataset,
        )
        self._agent = agent  # Store the agent
        
    def _single_deterministic_pass_dataset(
        self,
        sample_batch_size=None,
        num_steps=None,
        sequence_preprocess_fn=None,
        num_parallel_calls=None,
    ):
        """
        Overrides the original _single_deterministic_pass_dataset method to include preprocessing.

        Returns:
            A tf.data.Dataset containing all the preprocessed items in the buffer.
        """
        # Step 1: Gather raw data using the parent class's _gather_all
        raw_dataset = super(CustomTFUniformReplayBuffer, self)._single_deterministic_pass_dataset(
          sample_batch_size=sample_batch_size, 
          num_steps=num_steps, 
          sequence_preprocess_fn=None, 
          num_parallel_calls=num_parallel_calls)

        def per_sequence_fn(data, buffer_info):
            # At this point, each sample data contains a sequence of trajectories.
            preprocess_sequence_fn = common.function_in_tf1()(
                self._agent._preprocess_sequence
            )
            data = preprocess_sequence_fn(data)
            return data, buffer_info

        dataset = raw_dataset.map(per_sequence_fn, num_parallel_calls=num_parallel_calls)
        return dataset.prefetch(tf.data.AUTOTUNE)

#################################################################################
#                Define train_eval Wrapper for multiprocessing                  #
#################################################################################
def create_collect_env():
    collect_env = load_environment(collect_scenario_config)
    collect_env._metrics_path = None
    collect_env._occupancy_normalization_constant = 125.0
    return collect_env

def train_eval(
    # Training params
    num_iterations=20,
    actor_fc_layers=(128, 128),
    value_fc_layers=(128, 64),
    actor_learning_rate=3e-4,
    minibatch_size=504, # 4032(steps per episode) / 8
    num_epochs=10,
    # Agent params
    importance_ratio_clipping=0.2,
    lambda_value=0.95,
    discount_factor=0.99,
    entropy_regularization=0.0,
    value_pred_loss_coef=0.5,
    use_gae=True,
    use_td_lambda_return=True,
    gradient_clipping=0.5,
    value_clipping=None,
    # Replay params
    reverb_port=None,
    replay_capacity=10000,
    # Others
    policy_save_interval=1,
    summary_interval=1,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    num_parallel_environments=5,
):
    collect_env = ParallelPyEnvironment(
        [create_collect_env] * num_parallel_environments
    )
    
    eval_env = load_environment(eval_scenario_config)
    eval_env._metrics_path = metrics_path
    eval_env._occupancy_normalization_constant = 125.0

    observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(collect_env)
    train_step = train_utils.create_train_step()
    
    # @title Construct the PPO agent
    actor_net = ppo_actor_network.PPOActorNetwork().create_sequential_actor_net(
        fc_layer_units=actor_fc_layers,
        action_tensor_spec=action_spec,
    )

    value_net = value_network.ValueNetwork(
        input_tensor_spec=observation_spec,
        fc_layer_params=value_fc_layers,
        activation_fn=tf.keras.activations.relu
    )

    agent = ppo_clip_agent.PPOClipAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        optimizer=tf.keras.optimizers.Adam(learning_rate=actor_learning_rate),
        actor_net=actor_net,
        value_net=value_net,
        importance_ratio_clipping=importance_ratio_clipping,
        lambda_value=lambda_value,
        discount_factor=discount_factor,
        entropy_regularization=entropy_regularization,
        # epochs handled in learner
        num_epochs=1,
        use_gae=use_gae,
        use_td_lambda_return=use_td_lambda_return,
        gradient_clipping=gradient_clipping,
        value_clipping=value_clipping,
        value_pred_loss_coef=value_pred_loss_coef,
        # Skips updating normalizers in the agent, as it's handled in the learner.
        update_normalizers_in_train=False,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step,
        compute_value_and_advantage_in_train=False # when minibatch_size is used
    )
    agent.collect_policy._clip=True # change clip parameter from TFPolicy parent class to clip action tensor
    agent.initialize()
    
    sequence_length = int(eval_env.steps_per_episode)
    replay_capacity = num_parallel_environments * sequence_length
    
    
    # @title Set up the replay buffer
    train_replay_buffer = CustomTFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=num_parallel_environments,
        max_length=replay_capacity,
        agent=agent
    )
    
    normalization_replay_buffer = CustomTFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=num_parallel_environments,
        max_length=replay_capacity,
        agent=agent
    )
    
    def training_dataset_fn():
        return train_replay_buffer.as_dataset(
            sample_batch_size=num_parallel_environments,
            num_steps=sequence_length,
            single_deterministic_pass=True
        )

    def normalization_dataset_fn():
        return train_replay_buffer.as_dataset(
            sample_batch_size=num_parallel_environments,
            num_steps=sequence_length,
            single_deterministic_pass=True
        )

    saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
    print('Policies will be saved to saved_model_dir: %s' %saved_model_dir)
    env_step_metric = py_metrics.EnvironmentSteps()
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            agent,
            train_step,
            interval=policy_save_interval,
            metadata_metrics={triggers.ENV_STEP_METADATA_KEY: env_step_metric},
        ),
        triggers.StepPerSecondLogTrigger(train_step, interval=10),
    ]

    # @title Define an Agent Learner
    agent_learner = ppo_learner.PPOLearner(
        root_dir,
        train_step,
        agent,
        experience_dataset_fn=training_dataset_fn,
        normalization_dataset_fn=normalization_dataset_fn,
        num_samples=1,
        num_epochs=num_epochs,
        triggers=learning_triggers,
        minibatch_size=minibatch_size,
        shuffle_buffer_size=sequence_length,
        summary_interval=summary_interval
    )

    # @title Define Observers
    collect_print_status_observer = PrintStatusObserver(
        status_interval_steps=1,
        environment=collect_env.envs[0],
        replay_buffer=train_replay_buffer,
        num_timesteps_in_episode=eval_env.steps_per_episode
    )

    eval_render_plot_observer = RenderAndPlotObserver(
        render_interval_steps=144, environment=eval_env
    )
    
    eval_print_status_observer = PrintStatusObserver(
        status_interval_steps=1, environment=eval_env, replay_buffer=train_replay_buffer
    )

    tf_collect_policy = agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True
    )

    collect_actor = actor.Actor(
        collect_env,
        collect_policy,
        train_step,
        steps_per_run=sequence_length * num_parallel_environments,
        metrics=actor.collect_metrics(1)+ [env_step_metric],
        reference_metrics=[env_step_metric],
        summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
        summary_interval=1,
        observers=[
            train_replay_buffer.add_batch,
            normalization_replay_buffer.add_batch,
            env_step_metric,
            collect_print_status_observer,
            # collect_render_plot_observer,
        ],
    )

    tf_greedy_policy = greedy_policy.GreedyPolicy(agent.policy)
    eval_greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_greedy_policy, use_tf_function=True
    )

    eval_actor = actor.Actor(
        eval_env,
        eval_greedy_policy,
        train_step,
        episodes_per_run=1,
        metrics=actor.eval_metrics(1),
        reference_metrics=[env_step_metric],
        summary_dir=os.path.join(root_dir, 'eval'),
        summary_interval=1,
        observers=[eval_print_status_observer, eval_render_plot_observer],
    )
    
   
    # training loop
    for iter in range(1, num_iterations+1):
        print('Training iteration: ', iter)
        # Let the collect actor run, using its stochastic action selection policy.
        logging_info("Collecting.")
        collect_actor.run()
        logging_info(
            'Executing gradient updates with %d frames.'
            %int(train_replay_buffer.num_frames())
        )
        # Now, with the additional collectsteps in the replay buffer,
        # allow the agent to make additional policy improvements.
        
        loss_info = agent_learner.run()
        logging_info(
            'Policy Gradient Loss: %6.2f, Value Estimation Loss: %6.2f, Clip Fraction: %6.2f '
            % (
                loss_info.extra.policy_gradient_loss.numpy(),
                loss_info.extra.value_estimation_loss.numpy(),
                loss_info.extra.clip_fraction.numpy(),
            )
        )
        # clearing buffer after training according to the PPO algorithm
        train_replay_buffer.clear()
        normalization_replay_buffer.clear()

        logging_info('Evaluating.')

        if iter % 4 == 0:
            _ = eval_env.reset()
            # Run the eval actor after the training iteration, and get its performance.
            eval_actor.run_and_log()


def main(_):
    train_eval(minibatch_size=504, num_iterations=10, num_epochs=10, num_parallel_environments=8)
    
    
if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(app.run, main))
  