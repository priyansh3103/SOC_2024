import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def q_learning(env, num_episodes, alpha=0.1, gamma=0.9):
    epsilon = 1/k
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    cumulative_rewards = []

    for i in range(num_episodes):
        state = env.reset()[0]  # Reset environment and get initial state
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # Q-learning update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state

        cumulative_rewards.append(total_reward)
    k = k+1

    return Q, cumulative_rewards

# Environment and Training
env = gym.make('Taxi-v3')
num_episodes = 3000
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
k= 1

q_learning_Q, q_learning_rewards = q_learning(env, num_episodes, alpha, gamma, epsilon)

# Plotting cumulative rewards
plt.plot(range(num_episodes), np.cumsum(q_learning_rewards), label='Q-learning')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.show()
