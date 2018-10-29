import os
import time
from collections import deque
import pickle
import queue
from threading import Thread

from baselines.ddpg2.ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI
import zmq

from baselines.ddpg2.replay_memory import RemoteReplayMemory
from baselines.ddpg2.replay_memory import Transition


class DDPGActor(object):

  def __init__(self, env, actor, critic, batch_size, memory_size,
               memory_warmup_size, gamma, tau, normalize_returns,
               normalize_observations, action_noise, param_noise, critic_l2_reg,
               actor_lr, critic_lr, popart, clip_norm, reward_scale,
               send_freq=4.0, ports=("6700", "6701", "6702"),
               learner_ip="localhost"):
    assert (np.abs(env.action_space.low) == env.action_space.high).all()
    self._env = env
    self._batch_size = batch_size
    self._max_action = env.action_space.high
    logger.info(
        'scaling actions by {} for executing in env'.format(self._max_action))
    self._agent = DDPG(actor,
                       critic,
                       env.observation_space.shape,
                       env.action_space.shape,
                       gamma=gamma,
                       tau=tau,
                       normalize_returns=normalize_returns,
                       normalize_observations=normalize_observations,
                       action_noise=action_noise,
                       param_noise=param_noise,
                       critic_l2_reg=critic_l2_reg,
                       actor_lr=actor_lr,
                       critic_lr=critic_lr,
                       enable_popart=popart,
                       clip_norm=clip_norm,
                       reward_scale=reward_scale)

    self._replay_memory = RemoteReplayMemory(
        is_server=False,
        memory_size=memory_size,
        memory_warmup_size=memory_warmup_size,
        send_freq=send_freq,
        ports=ports[:2],
        server_ip=learner_ip)

    self._zmq_context = zmq.Context()
    self._model_requestor = self._zmq_context.socket(zmq.REQ)
    self._model_requestor.connect("tcp://%s:%s" % (learner_ip, ports[2]))

  def run(self):
    with U.single_threaded_session() as sess:
      self._agent.initialize(sess)
      sess.graph.finalize()
      self._update_model()

      self._agent.reset()
      obs = self._env.reset()
      perturbed_distance = self._adapt_param_noise()

      cum_return, steps, episodes = 0.0, 0, 0
      start_time = time.time()
      while True:
        action, q = self._agent.pi(obs, apply_noise=True, compute_Q=True)
        assert action.shape == self._env.action_space.shape
        assert action.shape == self._max_action.shape
        new_obs, reward, done, info = self._env.step(self._max_action * action)
        transition = (obs, action, reward, new_obs, done)
        self._replay_memory.push(*transition)
        cum_return += reward
        steps += 1
        obs = new_obs
        if done:
          episodes += 1
          self._agent.reset()
          self._update_model()
          obs = self._env.reset()
          perturbed_distance = self._adapt_param_noise()
          print("Episode %d done. Return: %f. Steps: %f. "
              "Perturbed Distance: %f. Time: %f." % (episodes, cum_return,
              steps, perturbed_distance, time.time() - start_time))
          cum_return, steps = 0.0, 0
          start_time = time.time()

  def _adapt_param_noise(self):
    if self._replay_memory.total >= self._batch_size:
      batch = self._transitions_to_batch(
          self._replay_memory.sample(self._batch_size))
      return self._agent.adapt_param_noise(batch)
    else:
      return 0

  def _transitions_to_batch(self, transitions):
    batch = Transition(*zip(*transitions))
    observation = np.stack(batch.observation)
    next_observation = np.stack(batch.next_observation)
    action = np.stack(batch.action)
    reward = np.expand_dims(np.array(batch.reward, dtype=np.float32), axis=1)
    done = np.expand_dims(np.array(batch.done, dtype=np.float32), axis=1)
    return {
        'obs0': observation,
        'obs1': next_observation,
        'rewards': reward,
        'actions': action,
        'terminals1': done
    }

  def _update_model(self):
      self._model_requestor.send_string("request model")
      self._agent.load_params(self._model_requestor.recv_pyobj())


class DDPGLearner(object):

  def __init__(self, env, actor, critic, batch_size, memory_size,
               memory_warmup_size, gamma, tau, normalize_returns,
               normalize_observations, action_noise, param_noise, critic_l2_reg,
               actor_lr, critic_lr, popart, clip_norm, reward_scale,
               print_interval, ports=("6700", "6701", "6702")):
    self._batch_size = batch_size
    self._print_interval = print_interval
    self._ports = ports
    self._agent = DDPG(actor,
                       critic,
                       env.observation_space.shape,
                       env.action_space.shape,
                       gamma=gamma,
                       tau=tau,
                       normalize_returns=normalize_returns,
                       normalize_observations=normalize_observations,
                       action_noise=action_noise,
                       param_noise=param_noise,
                       critic_l2_reg=critic_l2_reg,
                       actor_lr=actor_lr,
                       critic_lr=critic_lr,
                       enable_popart=popart,
                       clip_norm=clip_norm,
                       reward_scale=reward_scale)
    self._replay_memory = RemoteReplayMemory(
        is_server=True,
        memory_size=memory_size,
        memory_warmup_size=memory_warmup_size,
        ports=ports[:2])

  def run(self):
    batch_queue = queue.Queue(8)
    batch_thread = Thread(target=self._prepare_batch,
                          args=(batch_queue, self._batch_size,))
    batch_thread.start()

    updates, rollout_frames, critic_loss, actor_loss = 0, 0, [], []
    time_start = time.time()
    with U.single_threaded_session() as sess:
      self._agent.initialize(sess)
      sess.graph.finalize()
      self._agent.reset()
      self._model_params = self._agent.read_params()
      self._reply_model_thread = Thread(target=self._reply_model,
                                        args=(self._ports[2],))
      self._reply_model_thread.start()
      while True:
        updates += 1
        batch = batch_queue.get()
        cl, al = self._agent.train(batch)
        critic_loss.append(cl), actor_loss.append(al)
        self._model_params = self._agent.read_params()

        if updates % self._print_interval == 0:
          time_elapsed = time.time() - time_start
          train_fps = self._print_interval * self._batch_size / time_elapsed
          rollout_fps = (self._replay_memory.total - rollout_frames) / time_elapsed
          print("Update: %d	Train-fps: %.1f	Rollout-fps: %.1f	"
              "Critic Loss: %.5f	Actor Loss: %.5f	Time: %.1f" % (updates,
              train_fps, rollout_fps, np.mean(critic_loss), np.mean(actor_loss),
              time_elapsed))
          time_start, critic_loss, actor_loss = time.time(), [], []
          rollout_frames = self._replay_memory.total

        self._agent.update_target_net()

  def _prepare_batch(self, batch_queue, batch_size):
    while True:
      transitions = self._replay_memory.sample(batch_size)
      batch = self._transitions_to_batch(transitions)
      batch_queue.put(batch)

  def _transitions_to_batch(self, transitions):
    batch = Transition(*zip(*transitions))
    observation = np.stack(batch.observation)
    next_observation = np.stack(batch.next_observation)
    action = np.stack(batch.action)
    reward = np.expand_dims(np.array(batch.reward, dtype=np.float32), axis=1)
    done = np.expand_dims(np.array(batch.done, dtype=np.float32), axis=1)
    return {
        'obs0': observation,
        'obs1': next_observation,
        'rewards': reward,
        'actions': action,
        'terminals1': done
    }

  def _reply_model(self, port):
    zmq_context = zmq.Context()
    receiver = zmq_context.socket(zmq.REP)
    receiver.bind("tcp://*:%s" % port)
    while True:
      msg = receiver.recv_string()
      assert msg == "request model"
      receiver.send_pyobj(self._model_params)
