"""
Training script for DQN agent on Flappy Bird
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import
from environments.flappy_bird_env import FlappyBirdEnv
from agents.dqn_agent import DQNAgent
from utils.helpers import load_config, Logger, get_timestamp
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import load_config, Logger, get_timestamp
import numpy as np
import matplotlib.pyplot as plt


def train_dqn(config, num_episodes=5000, render=False):
    """Train DQN agent on Flappy Bird"""
    
    # Setup
    logger = Logger(config['paths']['logs'])
    logger.log("="*60)
    logger.log("DQN TRAINING START")
    logger.log("="*60)
    
    # Create environment
    env = FlappyBirdEnv(render_mode='human' if render else None)
    logger.log(f"Environment created: {env}")
    logger.log(f"Observation space: {env.observation_space}")
    logger.log(f"Action space: {env.action_space}")
    
    # Create agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config
    )
    logger.log(f"Agent created with {sum(p.numel() for p in agent.policy_net.parameters())} parameters")
    
    # Training metrics
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    losses = []
    
    best_score = 0
    best_reward = float('-inf')
    
    logger.log("\nStarting training...\n")
    
    # Training loop
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        steps = 0
        
        while not done:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            # Update target network
            if agent.steps_done % agent.target_update_freq == 0:
                agent.update_target_network()
            
            # Render if enabled
            if render:
                env.render()
            
            # Update
            state = next_state
            episode_reward += reward
            steps += 1
            agent.steps_done += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_scores.append(info['score'])
        episode_lengths.append(steps)
        
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Track best
        if info['score'] > best_score:
            best_score = info['score']
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Log progress
        if (episode + 1) % config['training']['log_frequency'] == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_score = np.mean(episode_scores[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            
            logger.log(
                f"Episode {episode+1:4d} | "
                f"Score: {info['score']:2d} | "
                f"Reward: {episode_reward:7.1f} | "
                f"Avg Score: {avg_score:5.2f} | "
                f"Avg Reward: {avg_reward:7.1f} | "
                f"Steps: {steps:4d} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )
        
        # Save model
        if (episode + 1) % config['training']['save_frequency'] == 0:
            save_path = os.path.join(config['paths']['models'], f'dqn_episode_{episode+1}.pth')
            os.makedirs(config['paths']['models'], exist_ok=True)
            agent.save(save_path)
            logger.log(f"Model saved at episode {episode+1}")
    
    # Save final model
    final_path = os.path.join(config['paths']['models'], 'dqn_final.pth')
    agent.save(final_path)
    
    # Close environment
    env.close()
    
    # Log final stats
    logger.log("\n" + "="*60)
    logger.log("TRAINING COMPLETE")
    logger.log("="*60)
    logger.log(f"Best Score: {best_score}")
    logger.log(f"Best Reward: {best_reward:.1f}")
    logger.log(f"Final Epsilon: {agent.epsilon:.3f}")
    logger.log(f"Total Steps: {agent.steps_done}")
    
    # Plot results
    plot_training_results(episode_rewards, episode_scores, losses, config)
    
    return agent, episode_rewards, episode_scores


def plot_training_results(rewards, scores, losses, config):
    """Plot training metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(1, 3, 1)
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    # Moving average
    window = 100
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f'{window}-Episode Average', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    
    # Plot scores
    plt.subplot(1, 3, 2)
    plt.plot(scores, alpha=0.3, label='Score')
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(scores)), moving_avg, label=f'{window}-Episode Average', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score (Pipes Passed)')
    plt.title('Game Scores')
    plt.legend()
    plt.grid(True)
    
    # Plot losses
    plt.subplot(1, 3, 3)
    if losses:
        plt.plot(losses, alpha=0.6)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(config['paths']['logs'], f'training_results_{get_timestamp()}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Training plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    # Load config
    config = load_config()
    
    # Train
    print("Starting DQN training...")
    print("This will take a while on your i3 laptop (expect 2-3 hours for 5000 episodes)")
    print("You can stop early with Ctrl+C and the model will still be usable\n")
    
    try:
        agent, rewards, scores = train_dqn(
            config=config,
            num_episodes=1000,  # Start with 1000 for testing
            render=False  # Set to True to watch training (slower)
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Partial model saved")