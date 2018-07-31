#!/usr/bin/env python3
import sys
import os
import time
from baselines import logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.bench import Monitor
from baselines.ppo3 import ppo3
from baselines.ppo3.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
import multiprocessing
import tensorflow as tf


def start_actor(env_id, num_timesteps, seed, policy):

  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  ncpu = multiprocessing.cpu_count()
  if sys.platform == 'darwin': ncpu //= 2
  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=ncpu,
                          inter_op_parallelism_threads=ncpu)
  config.gpu_options.allow_growth = True #pylint: disable=E1101
  tf.Session(config=config).__enter__()

  env = make_atari(env_id)
  env.seed(int(time.time() * 1000))
  env = wrap_deepmind(env, {})
  env = Monitor(env, logger.get_dir())
  policy = {'cnn' : CnnPolicy,
            'lstm' : LstmPolicy,
            'lnlstm' : LnLstmPolicy,
            'mlp': MlpPolicy}[policy]
  model = ppo3.Model(policy=policy,
                     ob_space=env.observation_space,
                     ac_space=env.action_space,
                     nbatch_act=1,
                     nbatch_train=256,
                     nsteps=128,
                     ent_coef=0.01,
                     vf_coef=0.5,
                     max_grad_norm=0.5)
  actor = ppo3.PPOActor(env, model, nsteps=128, gamma=0.99, lam=0.95,
                        learner_ip="localhost")
  actor.run()

  ppo3.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
             lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
             ent_coef=.01,
             lr=lambda f : f * 2.5e-4,
             cliprange=lambda f : f * 0.1,
             total_timesteps=int(num_timesteps * 1.1))


def start_learner(env_id, policy):
  ncpu = multiprocessing.cpu_count()
  if sys.platform == 'darwin': ncpu //= 2
  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=ncpu,
                          inter_op_parallelism_threads=ncpu)
  config.gpu_options.allow_growth = True #pylint: disable=E1101
  tf.Session(config=config).__enter__()

  env = make_atari(env_id)
  env = wrap_deepmind(env, {})
  env = Monitor(env, logger.get_dir())
  policy = {'cnn' : CnnPolicy,
            'lstm' : LstmPolicy,
            'lnlstm' : LnLstmPolicy,
            'mlp': MlpPolicy}[policy]
  learner = ppo3.PPOLearner(env=env,
                            policy=policy,
                            nsteps=128,
                            #lr=lambda f: f * 2.5e-4,
                            lr=2.5e-4,
                            #cliprange=lambda f: f * 0.1,
                            cliprange=0.1,
                            print_interval=16,
                            nupdates=160000, 
                            ent_coef=0.01,
                            batch_size=256,
                            queue_size=8)
  learner.run()


def main():
  parser = atari_arg_parser()
  parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='lstm')
  parser.add_argument('--job_name', help='Job name', choices=['learner', 'actor'], default='learner')
  args = parser.parse_args()
  logger.configure()

  if args.job_name == 'actor':
    start_actor(args.env,
                num_timesteps=args.num_timesteps,
                seed=args.seed,
                policy=args.policy)
  else:
    start_learner(args.env,
                  policy=args.policy)

if __name__ == '__main__':
  main()
