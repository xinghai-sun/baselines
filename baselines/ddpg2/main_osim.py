import argparse
import time
import os
import logging

import tensorflow as tf
from mpi4py import MPI
import gym
from osim.env import ProstheticsEnv

from baselines import logger, bench
from baselines.common.misc_util import (
  set_global_seeds,
  boolean_flag,
)
import baselines.ddpg2.training as training
from baselines.ddpg2.training import DDPGActor, DDPGLearner
from baselines.ddpg2.models import Actor, Critic
from baselines.ddpg2.memory import Memory
from baselines.ddpg2.noise import *


class SymmetricActionWrapper(gym.ActionWrapper):

  def __init__(self, env):
    super(SymmetricActionWrapper, self).__init__(env)
    self._offset = (env.action_space.high - env.action_space.low) / 2
    env.action_space.high -= self._offset
    env.action_space.low -= self._offset

  def action(self, action):
    return action - self._offset

  def reverse_action(self, action):
    return action + self._offset


class NumpyObservationWrapper(gym.ObservationWrapper):

  def observation(self, obs):
    return np.array(obs, dtype=np.float32)


def create_env(seed):
  env = ProstheticsEnv(visualize=False)
  env.change_model(model='3D', prosthetic=True, difficulty=0, seed=seed)
  env = SymmetricActionWrapper(env)
  env = NumpyObservationWrapper(env)
  return env


def run_actor_worker(args):
  seed = int(time.time() * 1000) % 2^32
  logger.info('seed={}, logdir={}'.format(seed, logger.get_dir()))
  tf.reset_default_graph()
  set_global_seeds(seed)

  # Create envs.
  env = create_env(seed)
  env = bench.Monitor(env, logger.get_dir())

  # Parse noise_type
  action_noise = None
  param_noise = None
  num_actions = env.action_space.shape[-1]
  for current_noise_type in args.noise_type.split(','):
    current_noise_type = current_noise_type.strip()
    if current_noise_type == 'none':
      pass
    elif 'adaptive-param' in current_noise_type:
      _, stddev = current_noise_type.split('_')
      param_noise = AdaptiveParamNoiseSpec(
          initial_stddev=float(stddev), desired_action_stddev=float(stddev))
    elif 'normal' in current_noise_type:
      _, stddev = current_noise_type.split('_')
      action_noise = NormalActionNoise(
          mu=np.zeros(num_actions), sigma=float(stddev) * np.ones(num_actions))
    elif 'ou' in current_noise_type:
      _, stddev = current_noise_type.split('_')
      action_noise = OrnsteinUhlenbeckActionNoise(
          mu=np.zeros(num_actions), sigma=float(stddev) * np.ones(num_actions))
    else:
      raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

  # Configure components.
  critic_model = Critic(layer_norm=args.layer_norm)
  actor_model = Actor(num_actions, layer_norm=args.layer_norm)

  worker = DDPGActor(env=env,
                     actor=actor_model,
                     critic=critic_model,
                     batch_size=args.batch_size,
                     memory_size=args.client_memory_size,
                     memory_warmup_size=args.client_memory_warmup_size,
                     gamma=args.gamma,
                     tau=args.tau,
                     normalize_returns=args.normalize_returns,
                     normalize_observations=args.normalize_observations,
                     action_noise=action_noise,
                     param_noise=param_noise,
                     critic_l2_reg=args.critic_l2_reg,
                     actor_lr=args.actor_lr,
                     critic_lr=args.critic_lr,
                     popart=args.popart,
                     clip_norm=args.clip_norm,
                     reward_scale=args.reward_scale,
                     send_freq=args.actor_send_freq,
                     learner_ip=args.learner_ip,
                     ports=args.ports.split(','))
  worker.run()
  env.close()


def run_learner_worker(args):
  seed = int(time.time() * 1000) % 2^32
  logger.info('seed={}, logdir={}'.format(seed, logger.get_dir()))
  tf.reset_default_graph()
  set_global_seeds(seed)

  # Create envs.
  env = create_env(seed)
  env = bench.Monitor(env, logger.get_dir())

  # Parse noise_type
  action_noise = None
  param_noise = None
  num_actions = env.action_space.shape[-1]
  for current_noise_type in args.noise_type.split(','):
    current_noise_type = current_noise_type.strip()
    if current_noise_type == 'none':
      pass
    elif 'adaptive-param' in current_noise_type:
      _, stddev = current_noise_type.split('_')
      param_noise = AdaptiveParamNoiseSpec(
          initial_stddev=float(stddev), desired_action_stddev=float(stddev))
    elif 'normal' in current_noise_type:
      _, stddev = current_noise_type.split('_')
      action_noise = NormalActionNoise(
          mu=np.zeros(num_actions), sigma=float(stddev) * np.ones(num_actions))
    elif 'ou' in current_noise_type:
      _, stddev = current_noise_type.split('_')
      action_noise = OrnsteinUhlenbeckActionNoise(
          mu=np.zeros(num_actions), sigma=float(stddev) * np.ones(num_actions))
    else:
      raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

  # Configure components.
  critic_model = Critic(layer_norm=args.layer_norm)
  actor_model = Actor(num_actions, layer_norm=args.layer_norm)

  worker = DDPGLearner(env=env,
                       actor=actor_model,
                       critic=critic_model,
                       batch_size=args.batch_size,
                       memory_size=args.server_memory_size,
                       memory_warmup_size=args.server_memory_warmup_size,
                       gamma=args.gamma,
                       tau=args.tau,
                       normalize_returns=args.normalize_returns,
                       normalize_observations=args.normalize_observations,
                       action_noise=action_noise,
                       param_noise=param_noise,
                       critic_l2_reg=args.critic_l2_reg,
                       actor_lr=args.actor_lr,
                       critic_lr=args.critic_lr,
                       popart=args.popart,
                       clip_norm=args.clip_norm,
                       reward_scale=args.reward_scale,
                       print_interval=args.print_interval,
                       ports=args.ports.split(','))
  worker.run()
  env.close()

def parse_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--env-id', type=str, default='HalfCheetah-v1')
  parser.add_argument('--job-name', type=str, default='actor')
  parser.add_argument('--learner-ip', type=str, default='localhost')
  parser.add_argument('--ports', type=str, default='6700,6701,6702')
  boolean_flag(parser, 'render-eval', default=False)
  boolean_flag(parser, 'layer-norm', default=True)
  boolean_flag(parser, 'render', default=False)
  boolean_flag(parser, 'normalize-returns', default=False)
  boolean_flag(parser, 'normalize-observations', default=True)
  parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--client-memory-size', type=int, default=50000)
  parser.add_argument('--client-memory-warmup-size', type=int, default=1000)
  parser.add_argument('--server-memory-size', type=int, default=1000000)
  parser.add_argument('--server-memory-warmup-size', type=int, default=50000)
  parser.add_argument('--actor-send-freq', type=float, default=4.0)
  parser.add_argument('--actor-lr', type=float, default=1e-4)
  parser.add_argument('--critic-lr', type=float, default=1e-3)
  boolean_flag(parser, 'popart', default=False)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--tau', type=float, default=0.01)
  parser.add_argument('--reward-scale', type=float, default=1.)
  parser.add_argument('--clip-norm', type=float, default=None)
  parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')
  parser.add_argument('--print_interval', type=int, default=1000)
  boolean_flag(parser, 'evaluation', default=False)
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()
  logger.configure()
  if args.job_name == "actor":
    run_actor_worker(args)
  elif args.job_name == "learner":
    run_learner_worker(args)
