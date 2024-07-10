import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def q_learning(env, num_episodes, alpha=0.1, gamma=0.9):
    k= 1
    epsilon = 0.1
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    cumulative_rewards = []

    for i in range(num_episodes):
        state = env.reset()[0]  # Reset environment and get initial state
        done = False
        total_reward = 0
        dummy = True
        while not done:
            if dummy:
                if np.random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample() #explore
                else:
                    action = np.argmax(Q[state])  # Exploit
                next_state, reward, done, truncated, info = env.step(action)
                # Q-learning update
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + gamma * Q[next_state][best_next_action]
                td_error = td_target - Q[state][action]
                Q[state][action] += alpha * td_error
                
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
            
            total_reward += reward

            # Q-learning update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state
            if k>100000:
                done = True

        cumulative_rewards.append(total_reward)

    return Q, cumulative_rewards

# Environment and Training
env = gym.make('Taxi-v3')
num_episodes = 3000
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate

q_learning_Q, q_learning_rewards = q_learning(env, num_episodes, alpha, gamma)
np.save('ql_q_values.npy', q_learning_Q)
print("Trained and saved Q-values.")

# Plotting cumulative rewards
plt.plot(range(num_episodes), (q_learning_rewards), label='Q-learning')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.show()
