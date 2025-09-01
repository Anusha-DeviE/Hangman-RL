from env.hangman_env import HangmanEnv
from agents.dqn_agent import DQNAgent
from stable_baselines3 import PPO

def evaluate_dqn(agent, env, episodes=100):
    total_rewards = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        total_rewards += episode_reward
    avg_reward = total_rewards / episodes
    print(f"DQN Agent average reward over {episodes} episodes: {avg_reward}")

def evaluate_ppo(model, env, episodes=100):
    total_rewards = 0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        total_rewards += episode_reward
    avg_reward = total_rewards / episodes
    print(f"PPO-LSTM Agent average reward over {episodes} episodes: {avg_reward}")

if __name__ == "__main__":
    env = HangmanEnv()
    
    # Load trained DQN agent (example path)
    dqn_agent = DQNAgent()
    # You need to implement loading model weights here
    
    # Load trained PPO agent
    ppo_model = PPO.load("ppo_lstm_hangman")
    
    evaluate_dqn(dqn_agent, env)
    evaluate_ppo(ppo_model, env)
