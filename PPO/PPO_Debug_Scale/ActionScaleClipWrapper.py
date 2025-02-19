import numpy as np
from tf_agents.environments import PyEnvironmentBaseWrapper
from tensorflow.python.util import nest

class ActionScaleAndClipWrapper(PyEnvironmentBaseWrapper):
    """Wraps an environment to scale and then clip actions based on spec before applying."""

    def __init__(self, env, scale=5.0, name='ActionScaleAndClipWrapper'):
        """
        Initializes the ActionScaleAndClipWrapper.

        Args:
            env: The environment to wrap.
            scale: The factor by which to scale actions.
            name: Optional name for the wrapper.
        """
        super(ActionScaleAndClipWrapper, self).__init__(env)
        self.scale = scale

    def _step(self, action):
        """
        Steps the environment after scaling and clipping the actions.

        Args:
            action: Action to take.

        Returns:
            The next time_step from the environment.
        """
        env_action_spec = self._env.action_spec()

        def _clip_to_spec(act_spec, act):
            """
            Clips an action component based on its action_spec.

            Args:
                act_spec: The action specification (with minimum and maximum).
                act: The action value to clip.

            Returns:
                The clipped action value.
            """
            # Handle cases where min or max is None
            if act_spec.minimum is None and act_spec.maximum is None:
                return act
            return np.clip(act * self.scale, act_spec.minimum, act_spec.maximum)

        # Apply clipping to all action components
        clipped_actions = nest.map_structure_up_to(
            env_action_spec, _clip_to_spec, env_action_spec, action
        )

        return self._env.step(clipped_actions)