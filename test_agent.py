"""
Test the trained DQN agent
Watch it play Flappy Bird!
"""

import pygame
from environments.flappy_bird_env import FlappyBirdEnv
from agents.dqn_agent import DQNAgent
from utils.helpers import load_config

# Load config
config = load_config()

# Create environment WITH rendering
env = FlappyBirdEnv(render_mode='human')

# Create agent
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    config=config
)

# Load trained model
agent.load('trained_models/dqn_final.pth')
agent.epsilon = 0.0  # No exploration, pure exploitation

print("="*60)
print("TESTING TRAINED AGENT")
print("="*60)
print("Watch the AI play Flappy Bird!")
print("The bird has learned from 1000 episodes of training\n")
print("Close the game window to stop\n")

try:
    episode = 0
    running = True
    
    while running and episode < 10:
        state, info = env.reset()
        done = False
        total_reward = 0
        
        while not done and running:
            # IMPORTANT: Handle Pygame events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break
            
            # Agent decides best action (NO randomness)
            action = agent.select_action(state, training=False)
            
            # Take step
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Render game
            env.render()
            
            total_reward += reward
        
        if running:
            episode += 1
            print(f"Episode {episode}/10: Score = {info['score']:2d} pipes | Total Reward = {total_reward:7.1f}")
    
    env.close()
    print("\n" + "="*60)
    print("Testing complete! Your AI can play Flappy Bird!")
    print("="*60)

except KeyboardInterrupt:
    env.close()
    print("\n\nTesting stopped by user")
except Exception as e:
    env.close()
    print(f"\n\nError: {e}")