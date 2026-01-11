"""
Flappy Bird Game Environment
Simple implementation for RL training
"""

import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Game Constants
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
PIPE_WIDTH = 52
PIPE_HEIGHT = 320
PIPE_GAP = 100
GROUND_HEIGHT = 112
GRAVITY = 1
FLAP_STRENGTH = -10
PIPE_VELOCITY = -4
FPS = 10

# Colors (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (135, 206, 250)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)


class Bird:
    """The player-controlled bird"""
    
    def __init__(self):
        self.x = 50
        self.y = 150
        self.velocity = -2
        self.width = BIRD_WIDTH
        self.height = BIRD_HEIGHT
    
    def flap(self):
        """Make the bird jump"""
        self.velocity = FLAP_STRENGTH
    
    def update(self):
        """Update bird position with gravity"""
        self.velocity += GRAVITY
        self.y += self.velocity
        
        # Prevent bird from going off screen top
        if self.y < 0:
            self.y = 0
            self.velocity = 0
    
    def get_rect(self):
        """Get collision rectangle"""
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def draw(self, screen):
        """Draw the bird as a yellow circle"""
        pygame.draw.circle(screen, YELLOW, 
                         (int(self.x + self.width/2), 
                          int(self.y + self.height/2)), 
                         self.width//2)


class Pipe:
    """Obstacle pipes that move left"""
    
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH
        
        # Randomize gap position
        self.gap_y = random.randint(150, SCREEN_HEIGHT - GROUND_HEIGHT - 150)
        
        # Top pipe ends at gap_y
        self.top_height = self.gap_y - PIPE_GAP // 2
        
        # Bottom pipe starts after gap
        self.bottom_y = self.gap_y + PIPE_GAP // 2
        self.bottom_height = SCREEN_HEIGHT - GROUND_HEIGHT - self.bottom_y
        
        self.passed = False  # Track if bird passed this pipe
    
    def update(self):
        """Move pipe to the left"""
        self.x += PIPE_VELOCITY
    
    def is_off_screen(self):
        """Check if pipe has moved off screen"""
        return self.x + self.width < 0
    
    def get_rects(self):
        """Get collision rectangles for top and bottom pipes"""
        top_rect = pygame.Rect(self.x, 0, self.width, self.top_height)
        bottom_rect = pygame.Rect(self.x, self.bottom_y, 
                                  self.width, self.bottom_height)
        return top_rect, bottom_rect
    
    def draw(self, screen):
        """Draw the pipes as green rectangles"""
        top_rect, bottom_rect = self.get_rects()
        pygame.draw.rect(screen, GREEN, top_rect)
        pygame.draw.rect(screen, GREEN, bottom_rect)


class FlappyBirdGame:
    """Main game logic"""
    
    def __init__(self, render_mode=True):
        self.render_mode = render_mode
        
        # Create screen only if rendering
        if self.render_mode:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird - RL Training")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None
        
        self.reset()
    
    def reset(self):
        """Reset game to initial state"""
        self.bird = Bird()
        self.pipes = [Pipe(SCREEN_WIDTH + 200)]
        self.score = 0
        self.game_over = False
        self.frame_count = 0
        
        return self.get_state()
    
    def get_state(self):
        """Get current game state for AI agent"""
        # Find the next pipe (first pipe to the right of bird)
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + pipe.width > self.bird.x:
                next_pipe = pipe
                break
        
        if next_pipe is None:
            next_pipe = self.pipes[0]
        
        # State: [bird_y, bird_velocity, horizontal_distance, vertical_distance]
        state = np.array([
            self.bird.y,
            self.bird.velocity,
            next_pipe.x - self.bird.x,
            next_pipe.gap_y - self.bird.y
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """
        Execute one game step
        action: 0 = do nothing, 1 = flap
        returns: (state, reward, done, info)
        """
        self.frame_count += 1
        
        # Action
        if action == 1:
            self.bird.flap()
        
        # Update bird
        self.bird.update()
        
        # Update pipes
        for pipe in self.pipes:
            pipe.update()
        
        # Add new pipe when needed
        if self.pipes[-1].x < SCREEN_WIDTH - 200:
            self.pipes.append(Pipe(SCREEN_WIDTH + 50))
        
        # Remove off-screen pipes
        self.pipes = [p for p in self.pipes if not p.is_off_screen()]
        
        # Check collisions and scoring
        reward = 0.1  # Small reward for staying alive
        done = False
        
        # Check if bird hit ground or ceiling
        if self.bird.y + self.bird.height >= SCREEN_HEIGHT - GROUND_HEIGHT - 10:
            reward = -1000
            done = True
            self.game_over = True
        
        if self.bird.y <= 0:
            reward = -1000
            done = True
            self.game_over = True
        
        # Check pipe collisions
        bird_rect = self.bird.get_rect()
        for pipe in self.pipes:
            top_rect, bottom_rect = pipe.get_rects()
            
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                reward = -1000
                done = True
                self.game_over = True
            
            # Check if bird passed pipe
            if not pipe.passed and pipe.x + pipe.width < self.bird.x:
                pipe.passed = True
                self.score += 1
                reward = 100  # Big reward for passing pipe
        
        # Get new state
        state = self.get_state()
        
        # Info dictionary
        info = {
            'score': self.score,
            'frame': self.frame_count
        }
        
        return state, reward, done, info
    
    def render(self):
        """Draw the game"""
        if not self.render_mode or self.screen is None:
            return
        
        # Background
        self.screen.fill(BLUE)
        
        # Pipes
        for pipe in self.pipes:
            pipe.draw(self.screen)
        
        # Bird
        self.bird.draw(self.screen)
        
        # Ground
        ground_rect = pygame.Rect(0, SCREEN_HEIGHT - GROUND_HEIGHT, 
                                   SCREEN_WIDTH, GROUND_HEIGHT)
        pygame.draw.rect(self.screen, GREEN, ground_rect)
        
        # Score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Update display
        pygame.display.flip()
        
        if self.clock:
            self.clock.tick(FPS)
    
    def close(self):
        """Clean up"""
        if self.render_mode:
            pygame.quit()


# Test the game with human controls
if __name__ == "__main__":
    game = FlappyBirdGame(render_mode=True)
    
    print("Controls: SPACE or UP to flap")
    print("Close window to exit")
    
    running = True
    while running:
        # Handle events
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_SPACE, pygame.K_UP]:
                 action = 1
                print(f"🐦 FLAP! Bird at y={game.bird.y}")
        
        # Step game
        state, reward, done, info = game.step(action)
        
        # Render
        game.render()
        
        # Check if game over
        if done:
            print(f"Game Over! Score: {info['score']}")
            print("Press R to restart or close window to exit")
            
            # Wait for restart or quit
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            game.reset()
                            waiting = False
    
    game.close()
    print("Thanks for playing!")