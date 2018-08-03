#!/usr/bin/env python3
import sys
import os
import time
import multiprocessing

import tensorflow as tf

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.cmd_util import atari_arg_parser
from baselines.ppo3 import ppo3
from baselines.ppo3.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
from baselines.bench import Monitor


def tf_config():
  ncpu = multiprocessing.cpu_count()
  if sys.platform == 'darwin': ncpu //= 2
  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=ncpu,
                          inter_op_parallelism_threads=ncpu)
  config.gpu_options.allow_growth = True
  tf.Session(config=config).__enter__()


def start_actor(env_id, policy):
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  tf_config()

  env = make_atari(env_id)
  env.seed(int(time.time() * 1000))
  env = Monitor(env, "/tmp/")
  env = wrap_deepmind(env, frame_stack=True)
  policy = {'cnn' : CnnPolicy,
            'lstm' : LstmPolicy,
            'lnlstm' : LnLstmPolicy,
            'mlp': MlpPolicy}[policy]
  actor = ppo3.PPOActor(env=env,
                        policy=policy,
                        unroll_length=128,
                        gamma=0.99,
                        lam=0.95,
                        learner_ip="localhost")
  actor.run()


def start_learner(env_id, policy):
  tf_config()

  env = make_atari(env_id)
  env = Monitor(env, "/tmp/")
  env = wrap_deepmind(env, frame_stack=True)
  policy = {'cnn' : CnnPolicy,
            'lstm' : LstmPolicy,
            'lnlstm' : LnLstmPolicy,
            'mlp': MlpPolicy}[policy]
  learner = ppo3.PPOLearner(env=env,
                            policy=policy,
                            unroll_length=128,
                            lr=2.5e-4,
                            clip_range=0.1,
                            batch_size=2,
                            print_interval=100)
  learner.run()


def main():
  parser = atari_arg_parser()
  parser.add_argument('--policy', help='Policy architecture',
                      choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
  parser.add_argument('--job_name', help='Job name',
                      choices=['learner', 'actor'], default='learner')
  args = parser.parse_args()

  if args.job_name == 'actor':
    start_actor(args.env, policy=args.policy)
  else:
    start_learner(args.env, policy=args.policy)


if __name__ == '__main__':
  main()
