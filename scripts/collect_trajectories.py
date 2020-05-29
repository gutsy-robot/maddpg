import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from experiments.train import make_env
from experiments.train import get_trainers, mlp_model
import h5py


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10, help="number of episodes to be recorded")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")

    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--exp-name", type=str, default="simple", help="name of the experiment")
    parser.add_argument("--rollout_dir", type=str, default="/serverdata/sid/rollouts/", help="rollouts")

    parser.add_argument("--load-dir", type=str, default="/serverdata/sid/particle_env_policies/",
                        help="directory in which training state and model are loaded")
    parser.add_argument("--display", action="store_true", default=True)

    # don't seem necessary
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--save-dir", type=str, default="/serverdata/sid/particle_env_policies/",
                        help="directory in which training state and model should be saved")
    return parser.parse_args()


def main(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        all_states = []
        all_actions = []
        all_rewards = []

        # Initialize
        U.initialize()
        U.load_state(arglist.load_dir)

        obs_n = env.reset()
        episode_step = 0

        print('Starting iterations...')
        for i in range(arglist.num_episodes):
            print("starting episode: ", i)
            done = False
            terminal = False
            episode_states = []
            episode_actions = []
            episode_rewards = []
            while not done and not terminal:
                # get action
                # print("shape of obs_n is: ", (obs_n))
                episode_states.append(obs_n)
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                episode_actions.append(action_n)

                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                episode_step += 1
                done = all(done_n)

                episode_rewards.append(rew_n)
                # print(done)
                terminal = (episode_step >= arglist.max_episode_len)
                obs_n = new_obs_n

                if done or terminal:
                    obs_n = env.reset()
                    episode_step = 0
                if arglist.display:
                    time.sleep(0.1)
                    env.render()

            all_states.append(np.array(episode_states))
            all_actions.append(np.array(episode_actions))
            all_rewards.append(np.array(episode_rewards))

        all_states = np.array(all_states)
        all_actions = np.array(all_actions)
        all_rewards = np.array(all_rewards)
        h5f = h5py.File(arglist.rollout_dir + arglist.exp_name + '.h5', 'w')
        h5f.create_dataset('states', data=all_states)
        h5f.create_dataset('actions', data=all_actions)
        h5f.create_dataset('rewards', data=all_rewards)

        h5f.close()


args = parse_args()
main(args)

traj_file = h5py.File(args.rollout_dir + args.exp_name + '.h5', 'r')
print("shape of states ", traj_file['states'])
print("shape of actions ", traj_file['actions'])
print("shape of rewards ", traj_file['rewards'])