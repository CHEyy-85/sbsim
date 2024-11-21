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
import sys
# Add the directory for smart_control to sys.path
grandparent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(grandparent_dir)
import tensorflow as tf

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.agents.ppo import ppo_actor_network
from tf_agents.networks import value_network

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
    num_parallel_environments=1,
):
    if num_parallel_environments <= 0:
        raise ValueError("num_parallel_environments must be greater than 0")
    elif num_parallel_environments > 1:
        raise ValueError("Please checkout PPO_Batch or PPO_Parallel for more environments")
    
    collect_env = load_environment(collect_scenario_config)
    # For efficency, set metrics_path to None
    collect_env._metrics_path = None
    collect_env._occupancy_normalization_constant = 125.0
    
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

    reverb_checkpoint_dir = root_dir + "PPO_reverb_checkpoint"
    reverb_port = None
    print('reverb_checkpoint_dir=%s' %reverb_checkpoint_dir)
    reverb_checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
        path=reverb_checkpoint_dir
    )

    reverb_server = reverb.Server(
        [
            reverb.Table(  # Replay buffer storing experience for training.
                name='training_table',
                sampler=reverb.selectors.Fifo(),
                remover=reverb.selectors.Fifo(),
                rate_limiter=reverb.rate_limiters.MinSize(1),
                max_size=replay_capacity,
                max_times_sampled=1,
            ),
            reverb.Table(  # Replay buffer storing experience for normalization.
                name='normalization_table',
                sampler=reverb.selectors.Fifo(),
                remover=reverb.selectors.Fifo(),
                rate_limiter=reverb.rate_limiters.MinSize(1),
                max_size=replay_capacity,
                max_times_sampled=1,
            ),
        ],
        port=reverb_port,
        checkpointer=reverb_checkpointer
    )

    logging_info('reverb_server_port=%d' %reverb_server.port)
    reverb_replay_train = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        sequence_length=sequence_length,
        table_name='training_table',
        server_address='localhost:{}'.format(reverb_server.port),
        max_cycle_length=1,
        rate_limiter_timeout_ms=1000,
    )
    reverb_replay_normalization = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        sequence_length=sequence_length,
        table_name='normalization_table',
        server_address='localhost:{}'.format(reverb_server.port),
        max_cycle_length=1,
        rate_limiter_timeout_ms=1000,
    )

    rb_observer = reverb_utils.ReverbTrajectorySequenceObserver(
        reverb_replay_train.py_client,
        ['training_table', 'normalization_table'],
        sequence_length=sequence_length,
        stride_length=sequence_length,
    )
    
    # @title Make a TF Dataset
    def training_dataset_fn():
        return reverb_replay_train.as_dataset(
            sample_batch_size=num_parallel_environments,
            sequence_preprocess_fn=agent.preprocess_sequence,
        )

    def normalization_dataset_fn():
        return reverb_replay_normalization.as_dataset(
            sample_batch_size=num_parallel_environments,
            sequence_preprocess_fn=agent.preprocess_sequence,
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
    
    # @title Define a TF-Agents Actor for collect and eval
    collect_render_plot_observer = RenderAndPlotObserver(
    render_interval_steps=144, environment=collect_env
    )
    collect_print_status_observer = PrintStatusObserver(
        status_interval_steps=1,
        environment=collect_env,
        replay_buffer=reverb_replay_train,
    )
    eval_render_plot_observer = RenderAndPlotObserver(
        render_interval_steps=144, environment=eval_env
    )
    eval_print_status_observer = PrintStatusObserver(
        status_interval_steps=1, environment=eval_env, replay_buffer=reverb_replay_train
    )
    
    tf_collect_policy = agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True
    )

    collect_actor = actor.Actor(
        collect_env,
        collect_policy,
        train_step,
        steps_per_run=sequence_length,
        metrics=actor.collect_metrics(1)+ [env_step_metric],
        reference_metrics=[env_step_metric],
        summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
        summary_interval=1,
        observers=[
            rb_observer,
            env_step_metric,
            collect_print_status_observer,
            collect_render_plot_observer,
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
            %int(reverb_replay_train.num_frames())
        )
        # Now, with the additional collectsteps in the replay buffer,
        # allow the agent to make additional policy improvements.
        rb_observer.reset(write_cached_steps=False)
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
        reverb_replay_train.clear()
        reverb_replay_normalization.clear()

        logging_info('Evaluating.')

        if iter % 4 == 0:
            _ = eval_env.reset()
            # Run the eval actor after the training iteration, and get its performance.
            eval_actor.run_and_log()
        
    logging_info('')
    rb_observer.close()
    reverb_server.stop()


if __name__ == "__main__":
    train_eval()