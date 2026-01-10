"""
Main entry point for the Hybrid AI Agent project
"""

import argparse
from utils.helpers import load_config, setup_directories, get_device, set_seed, Logger


def main():
    """Main function to run the project"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hybrid AI Agent - NLP + RL")
    parser.add_argument(
        '--mode', 
        type=str, 
        default='train', 
        choices=['train', 'eval', 'demo'],
        help='Mode: train (train agent), eval (evaluate agent), demo (play game)'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of episodes to train (overrides config)'
    )
    args = parser.parse_args()
    
    # Load configuration
    print("="*60)
    print("HYBRID AI AGENT - Natural Language + Reinforcement Learning")
    print("="*60)
    
    config = load_config(args.config)
    
    # Setup environment
    setup_directories(config)
    device = get_device()
    set_seed(42)
    
    # Initialize logger
    logger = Logger(config['paths']['logs'])
    
    # Log startup info
    logger.log(f"Starting {config['project']['name']} v{config['project']['version']}")
    logger.log(f"Mode: {args.mode}")
    logger.log(f"Device: {device}")
    
    # Run based on mode
    if args.mode == 'train':
        logger.log("="*60)
        logger.log("TRAINING MODE")
        logger.log("="*60)
        logger.log("Training agent - Not implemented yet")
        logger.log("Next steps:")
        logger.log("  1. Build game environment (Flappy Bird)")
        logger.log("  2. Create DQN agent")
        logger.log("  3. Build rule extraction system")
        logger.log("  4. Train hybrid agent")
        
    elif args.mode == 'eval':
        logger.log("="*60)
        logger.log("EVALUATION MODE")
        logger.log("="*60)
        logger.log("Evaluating agent - Not implemented yet")
        logger.log("This will test the trained agent's performance")
        
    elif args.mode == 'demo':
        logger.log("="*60)
        logger.log("DEMO MODE")
        logger.log("="*60)
        logger.log("Running demo - Not implemented yet")
        logger.log("This will show the agent playing the game")
    
    logger.log("="*60)
    logger.log("Program finished successfully")
    logger.log("="*60)


if __name__ == "__main__":
    main()