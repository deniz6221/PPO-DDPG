from agent import PPOAgent
import gymnasium as gym
import config


if __name__ == "__main__":
    max_timesteps = 500

    env = gym.make(config, max_episode_steps=max_timesteps, render_mode="human", terminate_when_unhealthy=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim)
    agent.load_checkpoint("checkpoints/ppo_10000.pth")

    rews = []


    state, _ = env.reset()
    episode_reward = 0
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