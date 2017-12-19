import os
import time
from Embedding import Embedding
from FocusedcrawlingEnv import FocusedcrawlingEnv
import argparse
import numpy as np
np.random.seed(1337)  # for reproducibility
import distutils.util
from tensorforce.agents import PPOAgent, DQNAgent, VPGAgent
from tensorforce.execution import Runner

#parse the parameters
parser = argparse.ArgumentParser(description = "Text game simulators.")
parser.add_argument("--name", default='uiuc', type=str, help="name of the experiment i.e. wikinav, uiuc, reddit or name of the text game, e.g. savingjohn, machineofdeath, fantasyworld")
parser.add_argument("--anchortxt", type=distutils.util.strtobool, default='true', help="whether to use anchor text or full webpage text for the actions")
parser.add_argument("--backaction", type=distutils.util.strtobool, default='false', help="let the agent go back to the root")
parser.add_argument("--vocab", type=int, default=50000, help="number of words in vocab")
parser.add_argument("--maxlen", type=int, default=1000, help="number of words in vocab")
parser.add_argument("--episodes", type=int, default=100, help="number of episodes to train")
parser.add_argument("--steps", type=int, default=100, help="number of steps in each episode")
parser.add_argument("--layers", type=int, default=2, help="number of layers")
parser.add_argument("--hidden", type=int, default=2000, help="number of neurons per layer")
parser.add_argument("--activation", type=str, default='relu', help="activation")
parser.add_argument("--batchsize", type=int, default=64, help="batch_size DQN only")
parser.add_argument("--epsilon", type=float, default=0.2, help="exploration epsilon greedy DQN only")
parser.add_argument("--logfile", type=str, default='pgnewaug.csv', help="logfile for plotting etc")
parser.add_argument("--verbose", type=distutils.util.strtobool, default='false')

args = parser.parse_args()


### PARAMETERS
NUM_EPISODE = args.episodes
NUM_STEPS = args.steps
GLOVE_DIR =  "glove.txt" #GoogleNews-vectors-negative300.bin" "glove.840B.300d.txt"
NUM_DIMENSIONS = 300
N_WORDS = args.vocab
n_layers = args.layers
hidden_size = args.hidden
activation =  args.activation
batch_size = args.batchsize
epsilon=args.epsilon
staart = time.time()
start = time.time()
# embedding = Embedding(GLOVE_DIR)
print('Loading embeddings:', str(time.time()-start))

'''
DATA_DIR = "."
env = FocusedcrawlingEnv(os.path.join(DATA_DIR, 'data/uiucdataset.hdf5'), os.path.join(DATA_DIR, 'data/uiucgoal.pickle'),
                                 args.anchortxt, args.backaction, args.verbose, args.maxlen, embedding=embedding)

'''
# Network as list of layers
network_spec = [
    dict(type='embedding', indices=6000, size=300),
    dict(type='flatten'),
    dict(type='dense', size=200, activation='relu'),
    dict(type='dense', size=200, activation='relu')
]

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    states_spec=dict(type='int', shape=(1,args.maxlen)),
    # states_spec=dict(type='int', shape=(1,1000)),
    actions_spec=dict(type='int', num_actions=685),
    network_spec=network_spec,
    batch_size=64,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-4
    )
)

"""agent = DQNAgent(
    states_spec=dict(type='int', shape=(1,args.maxlen)),
    actions_spec=dict(type='int', num_actions=65),
    network_spec=network_spec,
    batch_size=1,
    first_update=1,
    target_sync_frequency=5
)"""

"""agent = VPGAgent(
    states_spec=dict(type='int', shape=(1,args.maxlen)),
    actions_spec=dict(type='int', num_actions=65),
    network_spec=network_spec,
    batch_size=1,
    optimizer=dict(
        type='adam',
        learning_rate=1e-4
    )
)"""


# Create the runner
runner = Runner(agent=agent, environment=env)

# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=3000, max_episode_timesteps=200, episode_finished=episode_finished)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
