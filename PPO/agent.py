from model import Actor, Critic
import torch
import torch.optim as optim
import torch.nn.functional as F

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        self.log_probs = []
        self.rewards = []
        self.actions = []
        self.values = []
        self.dones = []
        self.states = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.actor(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        self.log_probs.append(log_prob)
        self.actions.append(action)
        self.values.append(self.critic(state))
        self.states.append(state)

        return action.squeeze(0).detach().numpy()
    
    def select_deterministic_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.actor(state)
        action = mean

        return action.squeeze(0).detach().numpy()
    
    def update_models(self):
        
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.cat(self.log_probs).detach()
        actions = torch.cat(self.actions).detach()
        values = torch.cat(self.values).detach()
        states = torch.cat(self.states).detach()

   

        advantages = returns - values.squeeze()

        for _ in range(10):
            new_mean, new_log_std = self.actor(states)
            new_std = torch.exp(new_log_std)
            new_dist = torch.distributions.Normal(new_mean, new_std)

            new_log_probs = new_dist.log_prob(actions).sum(dim=-1)
            ratio = (new_log_probs - log_probs).exp()

            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            new_values = self.critic(states)
            new_values = new_values.squeeze()

            loss_actor = -torch.min(surrogate1, surrogate2).mean()
            loss_critic = F.mse_loss(new_values.squeeze(), returns)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            loss_actor.backward()
            loss_critic.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.clear_memory()

    
    def clear_memory(self):
        del self.log_probs[:]
        del self.rewards[:]
        del self.actions[:]
        del self.values[:]
        del self.dones[:]
        del self.states[:]


    def save_checkpoint(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict()
                    }, path)
    

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def save_models(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()
                    }, path)
    
    def load_models(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])


