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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # -1 for cpu, currently available gpu 0, 1
import multiprocessing

# @title Imports
from absl import app
import functools
os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'true'
import tensorflow as tf

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.agents.ppo import ppo_actor_network
from tf_agents.environments import ParallelPyEnvironment
from tf_agents.networks import value_network

from tf_agents.metrics import py_metrics
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.system import system_multiprocessing as multiprocessing

from plotting_utils import *
from local_runtime_utils import *
from observer_classes import *
from env_utils import *

#################################################################################
#                Define train_eval Wrapper for multiprocessing                  #
#################################################################################
num_parallel_environments = 15

def create_collect_env():
    collect_env = load_environment(collect_scenario_config)
    collect_env._metrics_path = None
    collect_env._occupancy_normalization_constant = 125.0
    return collect_env

def train_eval():
    collect_env = ParallelPyEnvironment(
        [create_collect_env] * num_parallel_environments
    )
    
    eval_env = load_environment(eval_scenario_config)
    # eval_env._label += "_eval"
    eval_env._metrics_path = metrics_path
    eval_env._occupancy_normalization_constant = 125.0

    #######################################################
    #                   Define PPO Agent                  #
    #######################################################


    # Actor network fully connected layers.
    actor_fc_layers = (128, 128)

    # Value network observation fully connected layers.
    value_fc_layers = (128, 64)

    batch_size = 256
    actor_learning_rate = 3e-4

    # Replay params
    replay_capacity = 1000000
    debug_summaries = True
    summarize_grads_and_vars = True

    observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = spec_utils.get_tensor_specs(
        eval_env
    )

    actor_net = ppo_actor_network.PPOActorNetwork().create_sequential_actor_net(
        fc_layer_units=actor_fc_layers,
        action_tensor_spec=action_tensor_spec,
        seed=0
    )

    value_net = value_network.ValueNetwork(
        input_tensor_spec=observation_tensor_spec,
        fc_layer_params=value_fc_layers,
        activation_fn=tf.keras.activations.relu
    )

    train_step = train_utils.create_train_step()
    agent = ppo_clip_agent.PPOClipAgent(
        time_step_spec=time_step_tensor_spec,
        action_spec=action_tensor_spec,
        optimizer=tf.keras.optimizers.Adam(learning_rate=actor_learning_rate),
        actor_net=actor_net,
        value_net=value_net,
        importance_ratio_clipping=0.2,       # Example hyperparameter
        use_gae=True,
        entropy_regularization=0.01,
        num_epochs=25,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step,
    )
    agent.collect_policy._clip=True # change clip parameter from TFPolicy parent class to clip action tensor
    agent.initialize()

    # @title Set up the replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=num_parallel_environments,
        max_length=replay_capacity,
    )

    # @title set up checkpoints
    
    # @title Define Observers
    collect_print_status_observer = PrintStatusObserver(
        status_interval_steps=1,
        environment=collect_env.envs[0],
        replay_buffer=replay_buffer,
        num_timesteps_in_episode=eval_env.steps_per_episode
    )

    eval_render_plot_observer = RenderAndPlotObserver(
        render_interval_steps=144, environment=eval_env
    )
    eval_print_status_observer = PrintStatusObserver(
        status_interval_steps=1, environment=eval_env, replay_buffer=replay_buffer, num_timesteps_in_episode=eval_env.steps_per_episode
    )


    dataset = replay_buffer.as_dataset(
        single_deterministic_pass=True,
        num_steps=252, # 16 batches per episode - 4032 steps per episode
        # sample_batch_size=batch_size
    )
    dataset = dataset.shuffle(buffer_size=4050 * num_parallel_environments, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size=num_parallel_environments)


    # @title Define an Agent Learner
    policy_save_interval = 1 # Save the policy after every learning step.
    learner_summary_interval = 1 # Produce a summary of the critic, actor, and alpha losses after every gradient update step.
    experience_dataset_fn = lambda: dataset

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

    agent_learner = learner.Learner(
        root_dir,
        train_step,
        agent,
        experience_dataset_fn,
        triggers=learning_triggers,
        strategy=None,
        summary_interval=learner_summary_interval,
    )

    eval_greedy_policy = greedy_policy.GreedyPolicy(agent.policy)
    tf_greedy_policy = greedy_policy.GreedyPolicy(agent.policy)
    eval_greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_greedy_policy, use_tf_function=True
    )

    tf_collect_policy = agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True
    )

    collect_actor = actor.Actor(
        collect_env,
        collect_policy,
        train_step,
        # steps_per_run=252 * num_parallel_environments,
        episodes_per_run=num_parallel_environments,
        metrics=actor.collect_metrics(1),
        summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
        summary_interval=1,
        observers=[
            replay_buffer.add_batch,
            env_step_metric,
            collect_print_status_observer,
            # collect_render_plot_observer,
        ]
    )

    eval_actor = actor.Actor(
        eval_env,
        eval_greedy_policy,
        train_step,
        episodes_per_run=1,
        metrics=actor.eval_metrics(1),
        summary_dir=os.path.join(root_dir, 'eval'),
        summary_interval=1,
        observers=[
            eval_print_status_observer, 
            eval_render_plot_observer
        ],
    )

    num_episodes = 10
    num_gradient_updates_per_training_iteration = 16

    logging_info('Training.')
   
    for iter in range(num_episodes):
        print('Training iteration: ', iter)
        
        logging_info(f'Collecting.')
        _ = collect_env.reset()
        collect_actor.run()
        
        logging_info(
            'Executing %d gradient updates.'
            %num_gradient_updates_per_training_iteration
        )
        loss_info = agent_learner.run(
            iterations=num_gradient_updates_per_training_iteration
        )
        
        logging_info(
            'Policy Gradient Loss: %6.2f, Value Estimation Loss: %6.2f, Clip Fraction: %6.2f '
            % (
                loss_info.extra.policy_gradient_loss.numpy(),
                loss_info.extra.value_estimation_loss.numpy(),
                loss_info.extra.clip_fraction.numpy(),
            )
        )

        # logging_info('Evaluating.')
        # _ = eval_env.reset()
        # Run the eval actor after the training iteration, and get its performance.
        # eval_actor.run_and_log()
        replay_buffer.clear()


def load_parallel():
    collect_env = ParallelPyEnvironment(
        [create_collect_env] * 15
    )
    print('successfully loaded')
    collect_env.reset()
    print('successfully reloaded')

def main(_):
    train_eval()
    # load_parallel()
    

if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(app.run, main))