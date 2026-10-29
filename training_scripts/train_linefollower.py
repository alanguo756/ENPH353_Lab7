#!/usr/bin/env python3
import random
import gym_gazebo
import gymnasium as gym
import numpy as np
import pickle
from torch.utils.tensorboard.writer import SummaryWriter
from geometry_msgs.msg import Twist


def main():
    writer = SummaryWriter()
    env = gym.make('LineFollowerEnv-v0')

    observation_space = env.observation_space
    action_space = env.action_space

    epsilon = 0.8
    epsilon_decay = 0.99

    q_table = np.zeros((3, action_space.n))

    highest_reward = -float("inf")

    try:
        state, _ = env.reset()
        for episode in range(1000):
            total_reward = 0
            actions_per_episode = 0
            done = False

            while not done:
                actions_per_episode += 1
                action = choose_action(state, q_table, epsilon, action_space)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                q_table = learn(state, action, next_state, reward, q_table, terminated)

                state = next_state
                total_reward += reward

            # Log data
            print(f"Episode {episode}: Epsilon: {epsilon} | Total Reward: {total_reward}")
            writer.add_scalar("Total Reward", total_reward, episode)
            writer.add_scalar("Actions per episode", actions_per_episode, episode)

            if total_reward > highest_reward:
                highest_reward = total_reward
                print(f"New highest reward: {total_reward}.")
                saveQ(q_table, "best_q_table")

            state, _ = env.reset()
            epsilon = max(0.01, epsilon * epsilon_decay)  # Decay epsilon

    except KeyboardInterrupt:
        print("Training interrupted by user!")
    finally:
        env.close()
        writer.flush()
        writer.close()


def learn(state, action, next_state, reward, q_table, terminated):
    alpha = 0.1
    gamma = 0.9

    # No future value if the episode ended (next_state is -1, which is not a real row)
    best_next = 0.0 if terminated else np.max(q_table[next_state])
    td_target = reward + gamma * best_next
    q_table[state, action] += alpha * (td_target - q_table[state, action])
    
    return q_table


def choose_action(state, q_table, epsilon, action_space):
    if random.random() < epsilon:
        action = action_space.sample()          # explore
    else:
        action = int(np.argmax(q_table[state])) # exploit
    return action


def saveQ(q_table, save_path):
    # Pickle: exact copy of the table, used to reload and resume training
    with open(save_path + ".pkl", "wb") as f:
        pickle.dump(q_table, f)

    # CSV: human-readable, one row per state, one column per action
    np.savetxt(save_path + ".csv", q_table, delimiter=",",
               header="action_0_left,action_1_forward,action_2_right",
               comments="")


def loadQ(load_path):
    with open(load_path + ".pkl", "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    main()