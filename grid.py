import numpy as np
import matplotlib.pyplot as plt
import random

# Define the environment parameters
num_states = 16  # 4x4 grid
num_actions = 4  # actions: 0=up, 1=right, 2=down, 3=left
gamma = 0.9      # discount factor
alpha = 0.1      # learning rate
epsilon = 0.1    # exploration rate
episodes = 100  # total number of episodes

# Initialize the Q-table
Q = np.zeros((num_states, num_actions))

# Define the reward matrix (example with fixed rewards)
rewards = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10])
terminal_state = 15  # Goal state

# Function to choose an action based on epsilon-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(num_actions))  # explore
    else:
        return np.argmax(Q[state, :])  # exploit

# Function to get the next state based on the current state and action
def next_state(state, action):
    if action == 0 and state > 3:       # Up
        return state - 4
    elif action == 1 and (state + 1) % 4 != 0:  # Right
        return state + 1
    elif action == 2 and state < 12:    # Down
        return state + 4
    elif action == 3 and state % 4 != 0:  # Left
        return state - 1
    else:
        return state  # Invalid move, stay in the same state

# Function to plot the environment
def plot_grid(state, episode):
    grid = np.zeros((4, 4))
    grid[terminal_state // 4, terminal_state % 4] = 0.6  # Goal state
    grid[state // 4, state % 4] = 1  # Current state
    plt.imshow(grid, cmap='gray', interpolation='nearest')
    plt.title(f"Episode {episode}")
    plt.xticks([]), plt.yticks([])
    plt.pause(0.2)

# Training loop with visualization
plt.figure(figsize=(4, 4))
for episode in range(episodes):
    state = 0  # Start state
    while state != terminal_state:
        action = choose_action(state)
        new_state = next_state(state, action)
        reward = rewards[new_state]

        # Update Q-value
        Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
        state = new_state

        # Plot the current state
        if episode == 0 or episode == episodes - 1:
            plot_grid(state, episode)

plt.show()

# Print the learned Q-table
print("Learned Q-table:")
print(Q)
