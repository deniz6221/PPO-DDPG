from agent import DDPGAgent
import gymnasium as gym
import config
import json


if __name__ == "__main__":
    max_timesteps = 200
    env = gym.make(config.ENV_NAME, max_episode_steps=max_timesteps, terminate_when_unhealthy=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPGAgent(state_dim, action_dim)
    num_episodes = 20000
    

    rews = []


    for episode in range(1, num_episodes +1):
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
        
        rews.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward}")
        agent.update_models()
        
        if episode % 1000 == 0:
            with open(f"checkpoints/rews_{episode}.json", "w") as f:
                json.dump(rews, f)
            
            if episode % 5000 == 0:
                agent.save_checkpoint(f"checkpoints/ddpg_{episode}.pth")

    agent.save_models("models/models.pth")

    env.close()