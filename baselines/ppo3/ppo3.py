import os
import os.path as osp
import time
from collections import deque
from queue import Queue
from threading import Thread
import time
import random

import joblib
import numpy as np
import tensorflow as tf
import zmq

from baselines import logger
from baselines.common import explained_variance
from baselines.common.runners import AbstractEnvRunner


class Model(object):
  def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
               nsteps, ent_coef, vf_coef, max_grad_norm):
    sess = tf.get_default_session()

    act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
    train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)

    A = train_model.pdtype.sample_placeholder([None])
    ADV = tf.placeholder(tf.float32, [None])
    R = tf.placeholder(tf.float32, [None])
    OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
    OLDVPRED = tf.placeholder(tf.float32, [None])
    LR = tf.placeholder(tf.float32, [])
    CLIPRANGE = tf.placeholder(tf.float32, [])

    neglogpac = train_model.pd.neglogp(A)
    entropy = tf.reduce_mean(train_model.pd.entropy())

    vpred = train_model.vf
    vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
    vf_losses1 = tf.square(vpred - R)
    vf_losses2 = tf.square(vpredclipped - R)
    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
    ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
    pg_losses = -ADV * ratio
    pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
    clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
    loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
    with tf.variable_scope('model'):
      params = tf.trainable_variables()
    grads = tf.gradients(loss, params)
    if max_grad_norm is not None:
      grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
    grads = list(zip(grads, params))
    trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
    _train = trainer.apply_gradients(grads)

    def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
      advs = returns - values
      advs = (advs - advs.mean()) / (advs.std() + 1e-8)
      td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
      if states is not None:
        td_map[train_model.S] = states
        td_map[train_model.M] = masks
      return sess.run(
        [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
        td_map
      )[:-1]
    self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

    def save(save_path):
      ps = sess.run(params)
      joblib.dump(ps, save_path)

    def load(load_path):
      loaded_params = joblib.load(load_path)
      restores = []
      for p, loaded_p in zip(params, loaded_params):
        restores.append(p.assign(loaded_p))
      sess.run(restores)

    def read_params():
      return sess.run(params)

    def load_params(loaded_params):
      restores = []
      for p, loaded_p in zip(params, loaded_params):
        restores.append(p.assign(loaded_p))
      sess.run(restores)

    self.train = train
    self.train_model = train_model
    self.act_model = act_model
    self.step = act_model.step
    self.value = act_model.value
    self.initial_state = act_model.initial_state
    self.save = save
    self.load = load
    self.read_params = read_params
    self.load_params = load_params
    tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101


def constfn(val):
  def f(_):
    return val
  return f


def safemean(xs):
  return np.nan if len(xs) == 0 else np.mean(xs)


class PPOActor(object):

  def __init__(self, env, model, nsteps, gamma, lam,
               learner_ip="localhost", queue_size=8):
    self.env = env
    self.model = model
    self.nsteps = nsteps
    self.lam = lam
    self.gamma = gamma
    self.obs = np.zeros(env.observation_space.shape,
                        dtype=env.observation_space.dtype.name)
    self.obs[:] = env.reset()
    self.state = model.initial_state
    self.done = False
    self._model_updated = False

    self._zmq_context = zmq.Context()
    self._data_queue = Queue(queue_size)
    self._push_thread = Thread(target=self._push_data, args=(
        self._zmq_context, learner_ip, self._data_queue))
    self._push_thread.start()
    self._subscriber_thread = Thread(target=self._update_model,
                                     args=(self._zmq_context, learner_ip))
    self._subscriber_thread.start()

  def run(self):
    while not self._model_updated: time.sleep(1)
    while True:
      # TODO: try except
      rollout_data = self._nstep_rollout()
      self._data_queue.put(rollout_data)

  def _nstep_rollout(self):
    mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = \
        [],[],[],[],[],[]
    mb_states, epinfos = self.state, []
    for _ in range(self.nsteps):
      action, value, self.state, neglogpac = self.model.step(
          np.expand_dims(self.obs, 0), self.state, np.expand_dims(self.done, 0))
      mb_obs.append(self.obs.copy())
      mb_actions.append(action[0])
      mb_values.append(value[0])
      mb_neglogpacs.append(neglogpac[0])
      mb_dones.append(self.done)
      self.obs[:], reward, self.done, info = self.env.step(action)
      if self.done:
        self.obs[:] = self.env.reset()
      maybeepinfo = info.get('episode')
      if maybeepinfo: epinfos.append(maybeepinfo)
      mb_rewards.append(reward)
    mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
    mb_actions = np.asarray(mb_actions)
    mb_values = np.asarray(mb_values, dtype=np.float32)
    mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
    mb_dones = np.asarray(mb_dones, dtype=np.bool)
    last_values = self.model.value(
        np.expand_dims(self.obs, 0), self.state, np.expand_dims(self.done, 0))
    mb_returns = np.zeros_like(mb_rewards)
    mb_advs = np.zeros_like(mb_rewards)
    lastgaelam = 0
    for t in reversed(range(self.nsteps)):
      if t == self.nsteps - 1:
        nextnonterminal = 1.0 - self.done
        nextvalues = last_values[0]
      else:
        nextnonterminal = 1.0 - mb_dones[t + 1]
        nextvalues = mb_values[t + 1]
      delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - \
          mb_values[t]
      mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * \
          nextnonterminal * lastgaelam
    mb_returns = mb_advs + mb_values
    # TODO: check mb_state and self.state
    return (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs,
            mb_states, epinfos)

  def _push_data(self, zmq_context, learner_ip, data_queue):
    sender = zmq_context.socket(zmq.PUSH)
    sender.setsockopt(zmq.SNDHWM, 1)
    sender.setsockopt(zmq.RCVHWM, 1)
    sender.connect("tcp://%s:5700" % learner_ip)
    while True:
      data = data_queue.get()
      sender.send_pyobj(data)

  def _update_model(self, zmq_context, learner_ip):
    subscriber = zmq_context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.RCVHWM, 1)
    subscriber.connect("tcp://%s:5701" % learner_ip)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, u'model')
    while True:
      topic = subscriber.recv_string()
      self.model.load_params(subscriber.recv_pyobj())
      model_id = subscriber.recv_string()
      self._model_updated = True
      print("Model updated with id: %s" % model_id)


class PPOLearner(object):

  def __init__(self, env, policy, nsteps, lr, cliprange, nupdates,
               ent_coef, vf_coef=0.5, batch_size=256, queue_size=128,
               max_grad_norm=0.5, print_interval=100, save_interval=0,
               load_path=None):
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    assert batch_size % nsteps == 0, "batch_size should be times of nsteps."
    self._lr = lr
    self._cliprange=cliprange
    self._nupdates = nupdates
    self._batch_size = batch_size
    self._nsteps = nsteps
    self._print_interval = print_interval
    self._save_interval = save_interval
    self._episode_infos = deque(maxlen=200)

    self._model = Model(policy=policy,
                        ob_space=env.observation_space,
                        ac_space=env.action_space,
                        nbatch_act=1,
                        nbatch_train=batch_size,
                        nsteps=nsteps,
                        ent_coef=ent_coef,
                        vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm)
    if load_path is not None:
      self._model.load(load_path)
    self._data_need_split = True if self._model.initial_state is None else False

    self._zmq_context = zmq.Context()
    self._data_queue = deque(maxlen=queue_size)
    self._pull_thread = Thread(target=self._pull_data,
                               args=(self._zmq_context, self._data_queue,
                                     self._episode_infos))
    self._pull_thread.start()
    self._publish_thread = Thread(target=self._publish_model,
                                  args=(self._zmq_context, self._model))
    self._publish_thread.start()

  def run(self):
    while not self._can_sample_batch(self._batch_size):
      time.sleep(1)
    mblossvals = []
    tfirststart = time.time()
    tstart = time.time()
    for update in range(1, self._nupdates + 1):
      frac = 1.0 - (update - 1.0) / self._nupdates
      lrnow = self._lr(frac)
      cliprangenow = self._cliprange(frac)
      batch = self._sample_batch(self._batch_size)
      obs, returns, dones, actions, values, neglogpacs, states = (
          np.concatenate(arr) if arr[0] is not None else None
          for arr in zip(*batch))
      mblossvals.append(self._model.train(lrnow, cliprangenow, obs, returns,
                                          dones, actions, values, neglogpacs,
                                          states))
      if update % self._print_interval == 0:
        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(self._print_interval * self._batch_size / (tnow - tstart))
        ev = explained_variance(values, returns)
        logger.logkv("nupdates", update)
        logger.logkv("total_timesteps", update * self._batch_size)
        logger.logkv("fps", fps)
        logger.logkv("explained_variance", float(ev))
        logger.logkv('eprewmean',
                     safemean([info['r'] for info in self._episode_infos]))
        #print(self._episode_infos)
        logger.logkv('eplenmean',
                     safemean([info['l'] for info in self._episode_infos]))
        for (lossval, lossname) in zip(lossvals, self._model.loss_names):
          logger.logkv(lossname, lossval)
        logger.dumpkvs()
        mblossvals = []
        tstart = time.time()

      if (self._save_interval and update % self._save_interval == 0 and
          logger.get_dir()):
        checkdir = osp.join(logger.get_dir(), 'checkpoints')
        os.makedirs(checkdir, exist_ok=True)
        savepath = osp.join(checkdir, '%.5i'%update)
        print('Saving to', savepath)
        self._model.save(savepath)

  def _can_sample_batch(self, batch_size):
      assert batch_size % self._nsteps == 0
      return len(self._data_queue) >= self._batch_size // self._nsteps

  def _sample_batch(self, batch_size):
      assert batch_size % self._nsteps == 0
      need_shuffle = True if self._data_queue[0][-1] is None else False
      need_shuffle = False
      if need_shuffle:
        rand_ids = np.random.randint(len(self._data_queue) * self._nsteps,
                                     size=self._batch_size)
        queue_ids = rand_ids // self._nsteps
        sample_ids = rand_ids % self._nsteps
        return [tuple(arr[s:s+1] for arr in self._data_queue[q][:-1]) + (None,)
                for q, s in zip(queue_ids, sample_ids)]
      else:
        return random.sample(self._data_queue, batch_size // self._nsteps)

  def _pull_data(self, zmq_context, data_queue, episode_infos):
    receiver = zmq_context.socket(zmq.PULL)
    receiver.setsockopt(zmq.RCVHWM, 1)
    receiver.setsockopt(zmq.SNDHWM, 1)
    receiver.bind("tcp://*:5700")
    while True:
      data = receiver.recv_pyobj()
      for arr in data[:-2]: assert arr.shape[0] == self._nsteps
      data_queue.append(data[:-1])
      episode_infos.extend(data[-1])
      # TODO: compute fps

  def _publish_model(self, zmq_context, model):
    publisher = zmq_context.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 1)
    publisher.bind("tcp://*:5701")
    model_id = 0
    while True:
      publisher.send_string("model", zmq.SNDMORE)
      publisher.send_pyobj(model.read_params(), zmq.SNDMORE)
      publisher.send_string(str(model_id))
      model_id += 1
      time.sleep(2.0)
