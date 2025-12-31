import numpy as np
import matplotlib.pyplot as plt

# Environment
grid_size = 5
n_states = grid_size * grid_size
n_actions = 4  # up, down, left, right

# Q-table
Q_table = np.zeros((n_states, n_actions))

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
num_episodes = 500

# Rewards
rewards = np.full(n_states, -1)
rewards[24] = 10    # goal
rewards[12] = -10   # pitfall

# Epsilon-greedy policy
def epsilon_greedy_action(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])

# Training
rewards_q_learning = []

for episode in range(num_episodes):
    state = np.random.randint(n_states)
    total_reward = 0
    done = False

    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(n_states)
        reward = rewards[next_state]

        # Bellman update
        Q_table[state, action] += alpha * (
            reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action]
        )

        state = next_state
        total_reward += reward

        if next_state in {12, 24}:
            done = True

    rewards_q_learning.append(total_reward)

# Plot
plt.plot(rewards_q_learning)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Q-Learning Training Curve")
plt.show()

def moving_average(data, window=20):
    return np.convolve(data, np.ones(window)/window, mode='valid')

plt.plot(moving_average(rewards_q_learning))
plt.title("Smoothed Q-Learning Curve")
plt.show()

