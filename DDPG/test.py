from agent import DDPGAgent
import gymnasium as gym
import config


if __name__ == "__main__":
    max_timesteps = 500
    env = gym.make(config.ENV_NAME, max_episode_steps=max_timesteps, terminate_when_unhealthy=False, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPGAgent(state_dim, action_dim)
    agent.load_checkpoint("checkpoints/ddpg_15000.pth")

    state, _ = env.reset()
    episode_reward = 0
    for t in range(max_timesteps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        agent.replay_buffer.add((state, action, reward, next_state, terminated))

        state = next_state
        episode_reward += reward

        if terminated or truncated:
            break
    print(f"Episode Reward: {episode_reward}")
    
    env.close()