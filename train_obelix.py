import numpy as np
import torch
from obelix import OBELIX
from dqn_agent import DQNAgent
import os

def train():
    SCALING_FACTOR = 4
    ARENA_SIZE = 500
    MAX_STEPS = 1000
    NUM_EPISODES = 1500
    TARGET_UPDATE_STEPS = 2000
    SAVE_PATH = "obelix_ddqn_weights.pth"
    env = OBELIX(scaling_factor=SCALING_FACTOR, arena_size=ARENA_SIZE, max_steps=MAX_STEPS)
    agent = DQNAgent(state_dim=18, action_dim=5)
    ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
    print("Starting training...")

    total_steps = 0
    render_interval = 100  # Render every 100 episodes
    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action_idx = agent.select_action(state)
            action = ACTIONS[action_idx]
            
            next_state, reward, done = env.step(action, render=(episode + 1) % render_interval == 0)
            agent.step(state, action_idx, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps % TARGET_UPDATE_STEPS == 0:
                agent.update_target_network()
                print(f"Target network updated at step {total_steps}")
            
        print(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Total Steps: {total_steps}")
        
        # Periodically save weights
        if (episode + 1) % 10 == 0:
            agent.save(SAVE_PATH)
            print(f"Saved weights to {SAVE_PATH}")

    agent.save(SAVE_PATH)
    print("Training finished and model saved.")

if __name__ == "__main__":
    train()
