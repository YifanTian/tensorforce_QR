# from tensorforce.config import Configuration
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.agents import DQNAgent, PPOAgent
from tensorforce.execution import Runner
from Env import QueryReofrmulatorEnv
import numpy as np

# environment = MinimalTest(specification=[('int', ())])
DATA_DIR = '/home/yifantian/Downloads/QR_TF'
dset = 'train'
env = QueryReofrmulatorEnv(DATA_DIR, dset, is_train = True, verbose = True, reward = 'RECALL')
# environment = MinimalTest(specification=[('int', ())])

# network_spec = [
#     dict(type='dense', size=32)
# ]

# agent = DQNAgent(
#     states_spec=environment.states,
#     actions_spec=environment.actions,
#     network_spec=network_spec
# )

# network_spec = [
#     dict(type='embedding', indices=6000, size=300),
#     dict(type='flatten'),
#     dict(type='dense', size=200, activation='relu'),
#     dict(type='dense', size=200, activation='relu')
# ]

# network_spec = [
#     dict(type='embedding', indices=100, size=32),
#     dict(type='dense', size=32, activation='tanh'),
#     dict(type='dense', size=32, activation='tanh')
# ]

# agent = PPOAgent(
#    states_spec=dict(type='int', shape=(1,10)),
#    actions_spec=dict(type='int', num_actions=685),
# #    states_spec=environment.states,
# #    actions_spec=environment.actions,
#    batch_size=64,
#    step_optimizer=dict(
#        type='adam',
#        learning_rate=1e-4
#    ),
#       network_spec=network_spec
# )


# Network as list of layers
network_spec = [
    dict(type='embedding', indices=300, size=30),
    dict(type='flatten'),
    # dict(type='dense', size=200, activation='relu'),
    dict(type='dense', size=100, activation='relu')
]

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    # states_spec=dict(type='int', shape=(1,args.maxlen)),
    # states_spec=dict(type='int', shape=(1,1000)),
    # actions_spec=dict(type='int', num_actions=685),
    states_spec=env.states,
    actions_spec=env.actions,
    network_spec=network_spec,
    batch_size=64,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-4
    )
)

# Create the runner
runner = Runner(agent=agent, environment=env)

# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=20, max_episode_timesteps=10, episode_finished=episode_finished)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
