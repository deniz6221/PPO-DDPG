from agent import PPOAgent
import gymnasium as gym
import config
import time


if __name__ == "__main__":
    max_timesteps = 500

    env = gym.make(config.ENV_NAME, max_episode_steps=max_timesteps, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim)
    agent.load_models("models/models.pth")

    rews = []


    state, _ = env.reset()
    episode_reward = 0
    time.sleep(1)
    for t in range(max_timesteps):
        action = agent.select_deterministic_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        
        state = next_state
        episode_reward += reward

        if terminated or truncated:
            print(f"Episode finished after {t+1} timesteps")
            break
    
    print(f"Episode Reward: {episode_reward}")
               
    env.close()