# PPO3 (Distributed PPO with ZeroMQ)

## Introduction

- A distributed version of PPO with pyzmq.
- Original paper: https://arxiv.org/abs/1707.06347

## Example: Run Atari

### Actor(s)
```sh
for i in $(seq 0 7); do
  python3 -m baselines.ppo3.run_atari --job_name actor &
done;
wait
```

### Learner
```sh
python -m baselines.ppo3.run_atari --job_name learner
```
