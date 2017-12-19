# from tensorforce.config import Configuration
from tensorforce.environments.minimal_test import MinimalTest
from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner

environment = MinimalTest(specification=[('int', ())])

network_spec = [
    dict(type='dense', size=32)
]

# config = Configuration(
#     memory=dict(
#         type='replay',
#         capacity=1000
#     ),
#     batch_size=8,
#     first_update=100,
#     target_sync_frequency=50
# )

agent = DQNAgent(
    states_spec=environment.states,
    actions_spec=environment.actions,
    network_spec=network_spec
    # config=config
)
runner = Runner(agent=agent, environment=environment)

def episode_finished(runner):
    if runner.episode % 100 == 0:
        print(sum(runner.episode_rewards[-100:]) / 100)
    return runner.episode < 100 \
        or not all(reward >= 1.0 for reward in runner.episode_rewards[-100:])

runner.run(episodes=1000, episode_finished=episode_finished)
