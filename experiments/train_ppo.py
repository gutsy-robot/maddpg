import gym, argparse
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, PPO1
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from tag_positive_wrapper import TagPositiveWrapper
from train import get_trainers
import maddpg.common.tf_util as U
import tensorflow as tf
from stable_baselines import DDPG
import time

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=3, help="number of adversaries")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")

    parser.add_argument("--exp-name", type=str, default="tag_positive", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/serverdata/sid/particle_env_policies/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")

    return parser.parse_args()


arglist = parse_args()
scenario = scenarios.load("simple_tag.py").Scenario()
# create world
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

num_adversaries = min(env.n, arglist.num_adversaries)
g1 = tf.Graph()

with U.single_threaded_session():
    with g1.as_default():
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        U.initialize()
        print("var list is: ", tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        time.sleep(100)
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_0")
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir

        for l in range(1, num_adversaries):
            var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_" + str(l))

        saver = tf.train.Saver(var_list=var_list)

        U.load_state(arglist.load_dir, saver=saver)

        obs_n = env.reset()
        action_n = [agent.action(obs) for agent, obs in zip(trainers[:-1], obs_n[:-1])]
        print("action_n is: ", action_n)

env = TagPositiveWrapper(env, adversary_policy=trainers, adversary_graph=g1)
# print("agents are: ", env.agents[0])
# print("num  agents are: ", env.n)
print("observation space: ", env.observation_space)
print("action space: ", env.action_space)

model = PPO1(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=2500)
model.save("ppo_tag_positive")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_tag_positive")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()