import gymnasium as gym
import pygame
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_control(env, num_episodes, gamma=0.9, epsilon=0):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    returns_sum = {}
    returns_count = {}

    cumulative_rewards = []

    for i in range(num_episodes):
        state = env.reset()[0]  # Reset environment
        episode = []
        done = False
        total_reward = 0
        k = 0
        print("episode started")
        dummy = True
        # Generate an episode
        while not done:
            if dummy:
                if np.random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample() #explore
                else:
                    action = np.argmax(Q[state])  # Exploit
                next_state, reward, done, truncated, info = env.step(action)
                episode.append((state, action, reward))
                state = next_state
                total_reward += reward
                dummy = False
            k = k+1
            # Update state and action mask for the next state
            action_mask = info["action_mask"]
            valid_actions = np.where(action_mask == 1)[0]
            print(action_mask)
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(valid_actions)  # Explore from valid actions
                print(f"Exploring: Chose action {action} from valid actions {valid_actions} episode {i+1}")
            else:
                q_values = Q[state][valid_actions]
                max_q_value = np.max(q_values)
                best_actions = valid_actions[q_values == max_q_value]
                action = np.random.choice(best_actions)  # Exploit with random tie-breaking
                print(f"Exploiting: Chose action {action} from valid actions {valid_actions} with Q-values {q_values} episode {i+1}")
            next_state, reward, done, truncated, info = env.step(action)
            env.render()
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward
            if k>100000:
                done = True
            print(k)
        print("episode generated")
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
                print(f"state{state} action{action} Q_value{Q[state][action]}")
            #print("state-value function updated")
        print("episode complete")
    return Q, cumulative_rewards

# Environment and Training
env = gym.make('Taxi-v3', render_mode = "human")
num_episodes = 3000
gamma = 0.9
epsilon = 0

mc_Q, mc_rewards = monte_carlo_control(env, num_episodes, gamma, epsilon)
np.save('q_values.npy', mc_Q)
print("Trained and saved Q-values.")

# Plotting cumulative rewards
plt.plot(range(num_episodes), mc_rewards, label='Monte Carlo')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.show()
