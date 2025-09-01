import gym
from env.hangman_env import HangmanEnv
from agents.dqn_agent import DQNAgent
import numpy as np
from env.hangman_env import HangmanEnv
from agents.ppo_lstm_agent import create_ppo_lstm_agent


def train():
    env = HangmanEnv()
    agent = DQNAgent()
    episodes = 1000
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
        
        if ep % 10 == 0:
            print(f"Episode {ep}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

def train_ppo_lstm():
    env = HangmanEnv()
    model = create_ppo_lstm_agent(env)
    model.learn(total_timesteps=1000000)  # Adjust as needed
    model.save("ppo_lstm_hangman")


if __name__ == "__main__":
    train()

if __name__ == "__main__":
    train_ppo_lstm()