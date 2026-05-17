import os
import torch
import random
import numpy as np
from collections import deque
from pathlib import Path
from typing import List
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
STATE_SIZE = 13
MODEL_DIR= Path(__file__).parent.joinpath('model')
PTH_FILE = MODEL_DIR.joinpath('snake.pth')

class Agent:
    def __init__(self):
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Linear_QNet(STATE_SIZE, 256, 3)
        self.model.to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.n_games = 0

        if os.path.exists(PTH_FILE):
            checkpoint = torch.load(PTH_FILE, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.best_score = checkpoint['score']
        else:
            self.epoch = 0
            self.best_score = 0

    def get_basic_state(self, game: SnakeGameAI) -> List[bool]:
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        return [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

    @staticmethod
    def calculate_flood_fill(game: SnakeGameAI):
        """
        Returns a normalized score (0.0 to 1.0) representing
        how much of the board is accessible from (head_x, head_y).
        """
        # 1. Create a queue for BFS and a set for visited nodes
        queue = [game.head]
        visited = set([game.head])
        board_width = game.w
        board_height = game.h
        count = 0
        total_cells = board_width * board_height

        # 2. Run Standard BFS
        while queue:
            cx, cy = queue.pop(0)
            count += 1
            
            # Check all 4 neighbors
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                n = Point(cx + dx, cy + dy)
                if not game.is_collision(n) and n not in visited:
                    visited.add(n)
                    queue.append(n)

        # 3. Return normalized accessibility
        return count / total_cells

    def get_state(self, game: SnakeGameAI):
        state = list(map(float, self.get_basic_state(game)))
        total_squares = game.w * game.h

        # # --- NEW METRIC 1: FLOOD FILL (Avoid Traps) ---
        # # Run BFS to see how many squares are reachable from current head
        # # Returns int: e.g., 50 squares
        # reachable_count = self.calculate_flood_fill(game)
        # state.append(flood_fill_score)

        # # Normalize to 0-1 range
        # flood_fill_score = reachable_count / total_squares

        # --- NEW METRIC 2: TAIL DISTANCE (Loop Awareness) ---
        # Manhattan distance to the tail tip
        tail = game.snake[-1]
        dist_x = abs(game.head.x - tail.x)
        dist_y = abs(game.head.y - tail.y)
        manhattan_dist = dist_x + dist_y

        # Normalize by max possible distance (width + height)
        max_dist = game.w + game.h
        tail_score = manhattan_dist / max_dist
        state.append(tail_score)

        # --- NEW METRIC 3: SNAKE LENGTH (Growth Context) ---
        # Longer snakes need to play safer
        length_score = len(game.snake) / total_squares
        state.append(length_score)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.epoch
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_progress(self, score: int):
        if score <= self.best_score:
            return
        self.best_score = score
        checkpoint_to_save = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'score': score
        }
        torch.save(checkpoint_to_save, PTH_FILE)
        print('=' * 20)
        print(f"Saved progress, record score: {score}, loss: {self.trainer.current_loss}, accumulated games: {self.epoch}, games played this round: {self.n_games}")
        print('=' * 20)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.epoch += 1
            agent.train_long_memory()
            agent.save_progress(score)

            print('Game', agent.n_games, 'Score', score, 'Record:', agent.best_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    train()