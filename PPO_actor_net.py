# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Sequential Actor Network for PPO."""
import functools
import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.keras_layers import bias_layer
from tf_agents.networks import nest_map
from tf_agents.networks import sequential
from tf_agents.distributions import utils as distribution_utils
from tf_agents.specs import distribution_spec
from tf_agents.networks import utils as network_utils
from tf_agents.specs import tensor_spec


def tanh_and_scale_to_spec(inputs, spec):
  """Maps inputs with arbitrary range to range defined by spec using `tanh`."""
  means = (spec.maximum + spec.minimum) / 2.0
  magnitudes = (spec.maximum - spec.minimum) / 2.0

  return means + magnitudes * tf.tanh(inputs)


class PPOActorNetwork:
    """Contains the actor network structure."""
    def __init__(self, seed_stream_class=tfp.util.SeedStream):
        self.seed_stream_class = seed_stream_class

    def create_sequential_actor_net(
        self, fc_layer_units, action_tensor_spec, outer_rank, seed=None
    ):
        """Helper method for creating the actor network."""
        output_spec = self._output_distribution_spec(action_tensor_spec, "tanh_scaled_actor_net")
        
        self._seed_stream = self.seed_stream_class(
            seed=seed, salt='tf_agents_sequential_layers'
        )
        
        def _get_seed():
            seed = self._seed_stream()
            if seed is not None:
                seed = seed % sys.maxsize
            return seed

        def create_dist(loc_and_scale):
            loc = loc_and_scale['loc']
            loc = tanh_and_scale_to_spec(loc, action_tensor_spec)
            
            scale = loc_and_scale['scale']
            scale = tf.nn.softplus(scale)
            
            return output_spec.build_distribution(loc=loc, scale=scale)


        def means_layers():
            # TODO(b/179510447): align these parameters with Schulman 17.
            return tf.keras.layers.Dense(
                action_tensor_spec.shape.num_elements(),
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=0.1, seed=_get_seed()
                ),
                name='means_projection_layer',
            )

        def std_layers():
            # TODO(b/179510447): align these parameters with Schulman 17.
            # std_bias_initializer_value = np.log(np.exp(0.35) - 1)
            std_bias_initializer_value=0.0
            return bias_layer.BiasLayer(
                bias_initializer=tf.constant_initializer(
                    value=std_bias_initializer_value
                )
            )

        def no_op_layers():
            return tf.keras.layers.Lambda(lambda x: x)

        dense = functools.partial(
            tf.keras.layers.Dense,
            activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.Orthogonal(seed=_get_seed()),
        )

        return sequential.Sequential(
            [dense(num_units) for num_units in fc_layer_units]
            + [means_layers()]
            + [
                tf.keras.layers.Lambda(
                    lambda x: {'loc': x, 'scale': tf.zeros_like(x)}
                )
            ]
            + [
                nest_map.NestMap({
                    'loc': no_op_layers(),
                    'scale': std_layers(),
                })
            ]
            +
            # Create the output distribution from the mean and standard deviation.
            [tf.keras.layers.Lambda(create_dist)]
        )
    
    
    def _output_distribution_spec(self, sample_spec, network_name):
        is_multivariate = sample_spec.shape.ndims > 0
        param_properties = tfp.distributions.Normal.parameter_properties()
        input_param_spec = {  # pylint: disable=g-complex-comprehension
            name: tensor_spec.TensorSpec(
                shape=properties.shape_fn(sample_spec.shape),
                dtype=sample_spec.dtype,
                name=network_name + '_' + name,
            )
            for name, properties in param_properties.items()
        }

        def distribution_builder(*args, **kwargs):
            if is_multivariate:
                # For backwards compatibility, and because MVNDiag does not support
                # `param_static_shapes`, even when using MVNDiag the spec
                # continues to use the terms 'loc' and 'scale'.  Here we have to massage
                # the construction to use 'scale' for kwarg 'scale_diag'.  Since they
                # have the same shape and dtype expectationts, this is okay.
                kwargs = kwargs.copy()
                kwargs['scale_diag'] = kwargs['scale']
                del kwargs['scale']
                distribution = tfp.distributions.MultivariateNormalDiag(*args, **kwargs)
            else:
                distribution = tfp.distributions.Normal(*args, **kwargs)
                
            return distribution

        return distribution_spec.DistributionSpec(
            distribution_builder, input_param_spec, sample_spec=sample_spec
        )