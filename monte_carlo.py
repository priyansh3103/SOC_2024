import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_control(env, num_episodes, gamma=0.9, epsilon=0.1):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    returns_sum = {}
    returns_count = {}

    cumulative_rewards = []

    for i in range(num_episodes):
        state = env.reset()[0]  # Reset environment and get initial state
        episode = []
        done = False
        total_reward = 0

        # Generate an episode
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit

            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward

        cumulative_rewards.append(total_reward)

        # Calculate returns and update Q-values
        G = 0
        for t in range(len(episode)-1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            if not (state, action) in [(x[0], x[1]) for x in episode[0:t]]:
                if (state, action) not in returns_sum:
                    returns_sum[(state, action)] = 0.0
                    returns_count[(state, action)] = 0.0
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1.0
                Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

        # Update policy (epsilon-greedy)
        for state in range(env.observation_space.n):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

    return Q, cumulative_rewards

# Environment and Training
env = gym.make('Taxi-v3')
num_episodes = 3000
gamma = 0.9
epsilon = 0.1

mc_Q, mc_rewards = monte_carlo_control(env, num_episodes, gamma, epsilon)

# Plotting cumulative rewards
plt.plot(range(num_episodes), np.cumsum(mc_rewards), label='Monte Carlo')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.show()
