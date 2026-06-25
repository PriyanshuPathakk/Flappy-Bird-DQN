import flappy_bird_gymnasium
import gymnasium as gym
import torch
import itertools
import os
import argparse
import torch.nn as nn
import yaml
import random
import logging
from datetime import datetime
from dqn import DQN
import torch.optim as optim
from experience_replay import ReplayMemory

# Device Configuration Pipeline
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set
        
        # Ensure directories exist for artifacts
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Configure Logger
        log_filename = f"logs/dqn_{param_set}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
        )
        
        # Load hyperparameters from configuration
        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.env_id = params["env_id"]
        self.reward_thresh = params["reward_thresh"]

        # Exploration (Epsilon-Greedy) Parameters
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        # Deep Q-Network & Replay Buffer Hyperparameters
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]

        # Learning Parameters
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.loss_func = nn.MSELoss()
        self.optimizer = None

    def normalize_state(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float, device=device)
        # The observation space for FlappyBird is already normalized, dividing by 100 shrinks it to near-zero.
        return obs_t

    def run(self, is_training=True, render=False):            
        env = gym.make(self.env_id, render_mode="human" if render else None, use_lidar=False)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)
        best_reward = float('-inf')

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
        else:
            # If evaluating, load the best model checkpoint
            model_path = f"models/{self.param_set}_best.pt"
            if os.path.exists(model_path):
                policy_dqn.load_state_dict(torch.load(model_path, map_location=device))
                logging.info(f"Loaded best model checkpoint from {model_path} for evaluation.")
            else:
                logging.warning(f"No checkpoint found at {model_path}. Running unitialized policy.")
            epsilon = 0.0 # No exploration during evaluation

        logging.info(f"Starting execution loop. Mode: {'Training' if is_training else 'Evaluation'}")

        for episode in itertools.count():
            state, _ = env.reset()
            state = self.normalize_state(state)

            episode_reward = 0
            terminated = False

            while not terminated and episode_reward < self.reward_thresh:
                # Action Selection
                if is_training and random.random() < epsilon:
                    action_raw = env.action_space.sample()
                    action = torch.tensor(action_raw, dtype=torch.long, device=device) 
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).argmax(dim=-1).squeeze()
                
                next_state, reward, terminated, _, _ = env.step(action.item())
                
                # In flappy_bird_gymnasium, hitting a pipe returns -0.5 and the episode continues 
                # while the bird falls. We must terminate immediately so we don't train on dead frames.
                if reward < 0:
                    reward = -1.0
                    terminated = True
                    
                episode_reward += reward
                
                reward_t = torch.tensor(reward, dtype=torch.float, device=device)
                next_state_t = self.normalize_state(next_state)
                
                if is_training:
                    memory.append((state, action, next_state_t, reward_t, terminated))
                    steps += 1
                    
                    # Check sync rate on step steps rather than waiting for episode completions
                    if steps >= self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0

                    # Optimize network parameters
                    if len(memory) > self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                state = next_state_t

            # Logging metrics
            logging.info(f"Episode: {episode + 1} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.4f}")

            # Save Model Checkpoint if performance exceeds past historic highs
            if is_training and episode_reward > best_reward:
                best_reward = episode_reward
                model_path = f"models/{self.param_set}_best.pt"
                torch.save(policy_dqn.state_dict(), model_path)
                logging.info(f"New high score! Saved best model weights to {model_path}")

            # Decay Exploration Rate
            if is_training:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]
                
        # The actions tensor tensor dimensions now line up properly inside the gather layer
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_func(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Terminal Controller Mapping Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate a DQN Agent on Flappy Bird.")
    parser.add_argument("--config", type=str, default="flappybirdv0", help="Key name of parameter set inside parameters.yaml")
    parser.add_argument("--evaluate", action="store_true", help="Run the agent in evaluation mode using the saved best weights")
    parser.add_argument("--render", action="store_true", help="Enable graphic pygame rendering screen window")
    
    args = parser.parse_args()

    # Initialize and execute agent routine
    agent = Agent(param_set=args.config)
    agent.run(is_training=not args.evaluate, render=args.render)