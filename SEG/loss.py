
"""Seq2seq loss operations for use in sequence models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import tensorflow as tf

__all__ = ["sequence_loss","rl_loss"]


def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None):
  """Weighted cross-entropy loss for a sequence of logits.

  Depending on the values of `average_across_timesteps` and
  `average_across_batch`, the return Tensor will have rank 0, 1, or 2 as these
  arguments reduce the cross-entropy at each target, which has shape
  `[batch_size, sequence_length]`, over their respective dimensions. For
  example, if `average_across_timesteps` is `True` and `average_across_batch`
  is `False`, then the return Tensor will have shape `[batch_size]`.

  Args:
    logits: A Tensor of shape
      `[batch_size, sequence_length, num_decoder_symbols]` and dtype float.
      The logits correspond to the prediction across all classes at each
      timestep.
    targets: A Tensor of shape `[batch_size, sequence_length]` and dtype
      int. The target represents the true class at each timestep.
    weights: A Tensor of shape `[batch_size, sequence_length]` and dtype
      float. `weights` constitutes the weighting of each prediction in the
      sequence. When using `weights` as masking, set all valid timesteps to 1
      and all padded timesteps to 0, e.g. a mask returned by `tf.sequence_mask`.
    average_across_timesteps: If set, sum the cost across the sequence
      dimension and divide the cost by the total label weight across timesteps.
    average_across_batch: If set, sum the cost across the batch dimension and
      divide the returned cost by the batch size.
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A float Tensor of rank 0, 1, or 2 depending on the
    `average_across_timesteps` and `average_across_batch` arguments. By default,
    it has rank 0 (scalar) and is the weighted average cross-entropy
    (log-perplexity) per symbol.

  Raises:
    ValueError: logits does not have 3 dimensions or targets does not have 2
                dimensions or weights does not have 2 dimensions.
  """
  if len(logits.get_shape()) != 3:
    raise ValueError("Logits must be a "
                     "[batch_size x sequence_length x logits] tensor")
  if len(targets.get_shape()) != 2:
    raise ValueError("Targets must be a [batch_size x sequence_length] "
                     "tensor")
  if len(weights.get_shape()) != 2:
    raise ValueError("Weights must be a [batch_size x sequence_length] "
                     "tensor")
  with ops.name_scope(name, "sequence_loss", [logits, targets, weights]):
    num_classes = array_ops.shape(logits)[2]
    logits_flat = array_ops.reshape(logits, [-1, num_classes])
    targets = array_ops.reshape(targets, [-1])
    if softmax_loss_function is None:
      crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
          labels=targets, logits=logits_flat)
    else:
      crossent = softmax_loss_function(labels=targets, logits=logits_flat)
    final_crossent = crossent * array_ops.reshape(weights, [-1])
    # rewards = tf.tile(tf.expand_dims(rewards, 1),[1,array_ops.shape(logits)[1]])
    # final_crossent = mle_crossent * array_ops.reshape(rewards, [-1])
    if average_across_timesteps and average_across_batch:
      final_crossent = math_ops.reduce_sum(final_crossent)
      total_size = math_ops.reduce_sum(weights)
      total_size += 1e-12  # to avoid division by 0 for all-0 weights
      final_crossent /= total_size
    else:
      batch_size = array_ops.shape(logits)[0]
      sequence_length = array_ops.shape(logits)[1]
      final_crossent = array_ops.reshape(final_crossent, [batch_size, sequence_length])
    if average_across_timesteps and not average_across_batch:
      final_crossent = math_ops.reduce_sum(final_crossent, axis=[1])
      total_size = math_ops.reduce_sum(weights, axis=[1])
      total_size += 1e-12  # to avoid division by 0 for all-0 weights
      final_crossent /= total_size
    if not average_across_timesteps and average_across_batch:
      final_crossent = math_ops.reduce_sum(final_crossent, axis=[0])
      total_size = math_ops.reduce_sum(weights, axis=[0])
      total_size += 1e-12  # to avoid division by 0 for all-0 weights
      final_crossent /= total_size
    return final_crossent

def rl_loss(sample_dists, padding_mask,rewards):
  log_sample_dists = tf.log(sample_dists)
  masked_sample_dists = log_sample_dists * padding_mask
  prob_sample = tf.reduce_sum(masked_sample_dists, -1)
  rl_loss = rewards * (-prob_sample)
  return prob_sample, masked_sample_dists, tf.reduce_mean(rl_loss) # overall average