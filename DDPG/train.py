from agent import DDPGAgent
import gymnasium as gym
import config
import json


def custom_reward_function(reward, observation):
    # Make the agent go down by punishing its y coordinate
    return reward - observation[1] * 0.1

    

if __name__ == "__main__":
    max_timesteps = 500
    env = gym.make(config.ENV_NAME, max_episode_steps=max_timesteps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPGAgent(state_dim, action_dim)
    agent.load_checkpoint("checkpoints/ddpg_50000.pth")
    num_episodes = 80_000
    

    rews = json.load(open("checkpoints/rews_50000.json", "r")) 


    for episode in range(50_001, num_episodes +1):
        state, _ = env.reset()
        episode_reward = 0
        for t in range(max_timesteps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward = custom_reward_function(reward, next_state)
            
            agent.replay_buffer.add((state, action, reward, next_state, terminated))

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break
        
        rews.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward}")
        agent.update_models()
        
        if episode % 10000 == 0:
            with open(f"checkpoints/rews_custom_{episode}.json", "w") as f:
                json.dump(rews, f)
            
            agent.save_checkpoint(f"checkpoints/ddpg_custom_{episode}.pth")

    agent.save_models("models/models.pth")

    env.close()