"""Main experimental file."""

import os
import shutil
from absl import app
import pickle
from time import time

import numpy as np
import tensorflow as tf

from monte_carlo_gradients import bayes_lr
from monte_carlo_gradients import blr_model_grad_utils
from monte_carlo_gradients import config
from monte_carlo_gradients import control_variates
from monte_carlo_gradients import data_utils
from monte_carlo_gradients import dist_utils
from monte_carlo_gradients import gradient_estimators

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('estimator', type=str)
parser.add_argument('cv', type=str)
parser.add_argument('N', type=int, default=50)
args = parser.parse_args()

config.gradient_config.type = args.estimator
config.gradient_config.control_variate = args.cv if args.cv != "none" else ""
config.num_posterior_samples = args.N

config.experiment_dir = './results/' + \
  f'N-{config.num_posterior_samples}/' + \
  f'estimator-{config.gradient_config.type}/' + \
  f'cv-{config.gradient_config.control_variate if config.gradient_config.control_variate else "none"}/'


def _get_control_variate_fn():
  # Get control variate.
  control_variate = config.gradient_config.control_variate
  if control_variate == 'delta':
    return control_variates.control_delta_method
  if control_variate == 'moving_avg':
    return control_variates.moving_averages_baseline

  if control_variate:
    raise ValueError('Unsupported control variate')

  return None

def _get_grad_loss_fn():
  # Get gradient loss function.
  gradient_type_to_grad_loss_fn = {
      'pathwise': gradient_estimators.pathwise_loss,
      'score_function': gradient_estimators.score_function_loss,
      'measure_valued': gradient_estimators.measure_valued_loss,
  }

  return gradient_type_to_grad_loss_fn[config.gradient_config.type]

def _variance_reduction():
  return (config.gradient_config.control_variate or
          config.gradient_config.type == 'measure_valued')

def _jacobian_parallel_iterations():
  # The pathwise estimator requires more memory since it uses the backward
  # pass through the model, so we use less parallel iterations to compute
  # jacobians.
  return 100 if config.gradient_config.type == 'pathwise' else None


def _configure_hooks(train_loss):
  nan_hook = tf.train.NanTensorHook(train_loss)
  hooks = [nan_hook]
  return hooks


def main(argv):
  del argv

  # Training data.
  features, targets = data_utils.get_sklearn_data_as_tensors(
      batch_size=config.batch_size,
      dataset_name=config.dataset_name)

  # Eval data.
  eval_features, eval_targets = data_utils.get_sklearn_data_as_tensors(
      batch_size=None,
      dataset_name=config.dataset_name)
  dataset_size = eval_features.get_shape()[0]

  data_dims = features.shape[1]

  prior = dist_utils.multi_normal(
      loc=tf.zeros(data_dims), log_scale=tf.zeros(data_dims))

  with tf.variable_scope('posterior'):
    posterior = dist_utils.diagonal_gaussian_posterior(data_dims)

  model = bayes_lr.BayesianLogisticRegression(
      prior=prior, posterior=posterior,
      dataset_size=dataset_size,
      use_analytical_kl=config.use_analytical_kl)

  grad_loss_fn = _get_grad_loss_fn()
  control_variate_fn = _get_control_variate_fn()
  jacobian_parallel_iterations = _jacobian_parallel_iterations()

  def model_loss(features, targets, posterior_samples):
    num_posterior_samples_cv_coeff = config.num_posterior_samples_cv_coeff
    return blr_model_grad_utils.model_surrogate_loss(
        model,
        features, targets, posterior_samples,
        grad_loss_fn=grad_loss_fn,
        control_variate_fn=control_variate_fn,
        estimate_cv_coeff=config.estimate_cv_coeff,
        num_posterior_samples_cv_coeff=num_posterior_samples_cv_coeff,
        jacobian_parallel_iterations=jacobian_parallel_iterations)

  posterior_samples = posterior.sample(config.num_posterior_samples)
  train_loss, _ = model_loss(features, targets, posterior_samples)
  train_loss = tf.reduce_mean(train_loss)

  num_eval_posterior_samples = config.num_eval_posterior_samples
  eval_posterior_samples = posterior.sample(num_eval_posterior_samples)

  # Compute the surrogate loss without any variance reduction.
  # Used as a sanity check and for debugging.
  if _variance_reduction():
    if control_variate_fn:
      no_var_reduction_grad_fn = grad_loss_fn
    elif config.gradient_config.type == 'measure_valued':
      # Compute the loss and stats when not using coupling.
      def no_var_reduction_grad_fn(function, dist_samples, dist):
        return gradient_estimators.measure_valued_loss(
            function, dist_samples, dist, coupling=False)
    _, no_var_reduction_jacobians = blr_model_grad_utils.model_surrogate_loss(
        model, eval_features, eval_targets, eval_posterior_samples,
        grad_loss_fn=no_var_reduction_grad_fn,
        jacobian_parallel_iterations=jacobian_parallel_iterations)
  else:
    # No variance reduction used. No reason for additional logging.
    no_var_reduction_jacobians = {}

  for j in no_var_reduction_jacobians.values():
    assert j.get_shape().as_list()[0] == num_eval_posterior_samples

  start_learning_rate = config.start_learning_rate
  global_step = tf.train.get_or_create_global_step()

  if config.cosine_learning_rate_decay:
    training_steps = config.training_steps
    learning_rate_multiplier = tf.math.cos(
        np.pi / 2 * tf.cast(global_step, tf.float32)  / training_steps)
  else:
    learning_rate_multiplier = tf.constant(1.0)

  learning_rate = start_learning_rate * learning_rate_multiplier
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.minimize(train_loss, global_step=global_step)

  hooks = _configure_hooks(train_loss)

  with tf.train.MonitoredSession(hooks=hooks) as sess:
    times = list()
    for _ in range(config.training_steps):
      start = time()
      sess.run(train_op)
      runtime = time() - start
      times.append(runtime)

  if os.path.isdir(config.experiment_dir):
    shutil.rmtree(config.experiment_dir)
  os.makedirs(config.experiment_dir, exist_ok=False)
  with open(f'{config.experiment_dir}/times.pkl', 'wb') as f:
    pickle.dump(times, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

  app.run(main)