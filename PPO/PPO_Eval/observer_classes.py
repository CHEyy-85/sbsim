# Custom Observer Classes to Visualize Progress
import pandas as pd
import numpy as np
from tf_agents.trajectories import trajectory as trajectory_lib
from IPython.display import clear_output
from plotting_utils import render_env, plot_timeseries_charts
from env_utils import get_latest_episode_reader, time_zone

class RenderAndPlotObserver:
  """Renders and plots the environment."""

  def __init__(
      self,
      render_interval_steps: int = 10,
      environment=None,
  ):
    self._counter = 0
    self._render_interval_steps = render_interval_steps
    self._environment = environment
    self._cumulative_reward = 0.0

    self._start_time = None
    if self._environment is not None:
      self._num_timesteps_in_episode = (
          self._environment._num_timesteps_in_episode
      )
      self._environment._end_timestamp

  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:

    reward = trajectory.reward
    self._cumulative_reward += reward
    self._counter += 1
    if self._start_time is None:
      self._start_time = pd.Timestamp.now()

    if self._counter % self._render_interval_steps == 0 and self._environment:

      execution_time = pd.Timestamp.now() - self._start_time
      mean_execution_time = execution_time.total_seconds() / self._counter

      clear_output(wait=True)
      if self._environment._metrics_path is not None:
        reader = get_latest_episode_reader(self._environment._metrics_path)
        plot_timeseries_charts(reader, time_zone)

      render_env(self._environment)


class PrintStatusObserver:
  """Prints status information."""

  def __init__(
      self, status_interval_steps: int = 1, environment=None, replay_buffer=None
  ):
    self._counter = 0
    self._status_interval_steps = status_interval_steps
    self._environment = environment
    self._cumulative_reward = 0.0
    self._replay_buffer = replay_buffer

    self._start_time = None
    if self._environment is not None:
      self._num_timesteps_in_episode = (
          self._environment._num_timesteps_in_episode
      )
      self._environment._end_timestamp

  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:

    reward = trajectory.reward
    self._cumulative_reward += reward
    self._counter += 1
    if self._start_time is None:
      self._start_time = pd.Timestamp.now()

    if self._counter % self._status_interval_steps == 0 and self._environment:

      execution_time = pd.Timestamp.now() - self._start_time
      mean_execution_time = execution_time.total_seconds() / self._counter

      sim_time = self._environment.current_simulation_timestamp.tz_convert(
          time_zone
      )
      percent_complete = int(
          100.0 * (self._counter / self._num_timesteps_in_episode)
      )

      if self._replay_buffer is not None:
        rb_size = self._replay_buffer.num_frames()
        rb_string = " Replay Buffer Size: %d" % rb_size
      else:
        rb_string = ""

      print(
          "Step %5d of %5d (%3d%%) Sim Time: %s Reward: %2.2f Cumulative"
          " Reward: %8.2f Execution Time: %s Mean Execution Time: %3.2fs %s"
          % (
              self._environment._step_count,
              self._num_timesteps_in_episode,
              percent_complete,
              sim_time.strftime("%Y-%m-%d %H:%M"),
              reward,
              self._cumulative_reward,
              execution_time,
              mean_execution_time,
              rb_string,
          )
      )
      
if __name__ == "__main__":
    pass