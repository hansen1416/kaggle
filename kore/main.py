# All this syspath wranglig is needed to make sure that the agent runs on the target environment and can load both the external dependencies
# and the saved model. Dear kaggle, if possible, please make this easier!
import os
import sys

KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"
if os.path.exists(KAGGLE_AGENT_PATH):
    # We're in the kaggle target system
    sys.path.insert(0, os.path.join(KAGGLE_AGENT_PATH, 'lib'))
    agent_path = os.path.join(KAGGLE_AGENT_PATH, 'models', 'baseline_agent')
else:
    # We're somewhere else
    sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))
    agent_path = os.path.join('models', 'baseline_agent')

from stable_baselines3 import PPO  # nopep8
from environment import KoreGymEnv  # nopep8
# Now for the actual agent

model = PPO.load(agent_path)
kore_env = KoreGymEnv()


def agent(obs, config):
    kore_env.raw_obs = obs
    state = kore_env.obs_as_gym_state
    action, _ = model.predict(state)
    return kore_env.gym_to_kore_action(action)


if __name__ == "__main__":
    # This is for debugging purposes only before submitting - Are there any errors?
    from kaggle_environments import make
    from config import OPPONENT
    env = make("kore_fleets", debug=True)
    env.run(['main.py', OPPONENT])

    print(env)
