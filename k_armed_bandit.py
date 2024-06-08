import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import math

#chosing an arm based on maximum estimate and the epsilon probability
def index_epsilon_greedy(arr, epsilon):
    max_est = np.max(arr)
    indices_max_est = np.where(arr == max_est)[0]
    if len(indices_max_est) == 1:
        max_est_index = indices_max_est[0]
    else:
        max_est_index = random.choice(indices_max_est)

    prob = np.full(shape = 6,fill_value=epsilon/5)
    prob[max_est_index] = 1 - epsilon
    final_index = random.choice([0,1,2,3,4,5], p=prob)
    return final_index


estimates_1 = np.zeros(6)
estimates_2 = np.zeros(6)
estimates_3 = np.zeros(6)
occurence_1 = np.zeros(6)
occurence_2 = np.zeros(6)
occurence_3 = np.zeros(6)
reward_1 = 0
reward_2 = 0
reward_3 = 0
reward_episode_1 = np.empty(1000)
reward_episode_2 = np.empty(1000)
reward_episode_3 = np.empty(1000)
episode = 0
episode_x = np.zeros(1000)
time_step = 0

for i in range (1000):

    for i in range(100):
        #defining the arms
        arm_0 = random.normal(0, 1) # mean = 0 
        arm_1 =  random.choice([3, -4]) # mean = -0.5
        arm_2 = random.poisson(2) # mean = 2
        arm_3 = random.normal(1, 2) # mean = 1
        arm_4 = random.exponential(1) # mean = 1
        arm_5 = random.choice([random.normal(0,1),random.choice([3,-4]),random.poisson(2),random.normal(1,2),random.exponential(1)]) # mean = 0.7

        arms = np.array([arm_0,arm_1,arm_2,arm_3,arm_4,arm_5])

        #choosing an arm
        epsilon = 0.1
        arm_index = index_epsilon_greedy(estimates_1, epsilon)
        arm_value = arms[arm_index]
        #updating reward
        reward_1 += arm_value
        #updating occurence then estimates
        occurence_1[arm_index] += 1
        estimates_1[arm_index] = estimates_1[arm_index] + 1/occurence_1[arm_index] * (arm_value - estimates_1[arm_index])

        #choosing an arm
        epsilon = 0.01
        arm_index = index_epsilon_greedy(estimates_2, epsilon)
        arm_value = arms[arm_index]
        #updating reward
        reward_2 += arm_value
        #updating occurence then estimates
        occurence_2[arm_index] += 1
        estimates_2[arm_index] = estimates_2[arm_index] + 1/occurence_2[arm_index] * (arm_value - estimates_2[arm_index])

        #choosing an arm
        epsilon = 0
        arm_index = index_epsilon_greedy(estimates_3, epsilon)
        arm_value = arms[arm_index]
        #updating reward
        reward_3 += arm_value
        #updating occurence then estimates
        occurence_3[arm_index] += 1
        estimates_3[arm_index] = estimates_3[arm_index] + 1/occurence_3[arm_index] * (arm_value - estimates_3[arm_index])

        time_step += 1
    
    reward_episode_1[episode] = reward_1
    reward_episode_2[episode] = reward_2
    reward_episode_3[episode] = reward_3
    episode_x[episode] = episode + 1
    episode += 1

print(estimates_1)
print(occurence_1)
print(estimates_2)
print(occurence_2)
print(estimates_3)
print(occurence_3)



plt.plot(episode_x, reward_episode_1, color = 'r')
plt.plot(episode_x, reward_episode_2, color = 'y')
plt.plot(episode_x, reward_episode_3, color = 'b')
plt.title("k_armed_bandit")
plt.xlabel("episodes")
plt.ylabel("total reward after each label")
plt.grid()
plt.show()



