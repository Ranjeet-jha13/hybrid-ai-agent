"""
Deep Q-Network (DQN) Agent
Learns to play Flappy Bird using reinforcement learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    """
    Neural network that estimates Q-values for each action
    Input: game state (4 values)
    Output: Q-values for each action (2 values)
    """
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super(QNetwork, self).__init__()
        
        # Neural network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """Forward pass through the network"""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class ReplayBuffer:
    """
    Stores experiences and samples random batches for training
    Breaks correlation in training data
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent that learns to play the game
    Uses Q-learning with neural network function approximation
    """
    
    def __init__(self, state_dim=4, action_dim=2, config=None):
        # Device (CPU for your laptop)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Hyperparameters
        if config:
            self.hidden_dim = config['agent']['dqn']['hidden_dim']
            self.learning_rate = config['agent']['dqn']['learning_rate']
            self.gamma = config['agent']['dqn']['gamma']
            self.epsilon_start = config['agent']['dqn']['epsilon_start']
            self.epsilon_end = config['agent']['dqn']['epsilon_end']
            self.epsilon_decay = config['agent']['dqn']['epsilon_decay']
            self.batch_size = config['agent']['dqn']['batch_size']
            self.buffer_size = config['agent']['dqn']['replay_buffer_size']
            self.target_update_freq = config['agent']['dqn']['target_update_frequency']
        else:
            # Default values
            self.hidden_dim = 64
            self.learning_rate = 0.001
            self.gamma = 0.99
            self.epsilon_start = 1.0
            self.epsilon_end = 0.01
            self.epsilon_decay = 0.995
            self.batch_size = 32
            self.buffer_size = 10000
            self.target_update_freq = 1000
        
        # Networks
        self.policy_net = QNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained directly
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # Training state
        self.epsilon = self.epsilon_start
        self.steps_done = 0
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        With probability epsilon: random action (exploration)
        Otherwise: best action according to Q-network (exploitation)
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, 1)
        else:
            # Exploit: best action from Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step
        Sample batch from replay buffer and update Q-network
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        print(f"Model loaded from {filepath}")


# Test the agent
if __name__ == "__main__":
    print("Testing DQN Agent\n")
    
    # Create agent
    agent = DQNAgent(state_dim=4, action_dim=2)
    
    print(f"Policy Network:")
    print(agent.policy_net)
    print(f"\nTotal parameters: {sum(p.numel() for p in agent.policy_net.parameters())}")
    
    # Test forward pass
    dummy_state = np.array([100, 5, 200, 50], dtype=np.float32)
    action = agent.select_action(dummy_state)
    print(f"\nTest state: {dummy_state}")
    print(f"Selected action: {action}")
    
    # Test storing experience
    agent.store_experience(dummy_state, action, 1.0, dummy_state, False)
    print(f"\nReplay buffer size: {len(agent.replay_buffer)}")
    
    # Test training (need more data first)
    for _ in range(100):
        agent.store_experience(dummy_state, action, 1.0, dummy_state, False)
    
    loss = agent.train_step()
    print(f"Training loss: {loss}")
    
    print("\n✓ DQN Agent test complete!")