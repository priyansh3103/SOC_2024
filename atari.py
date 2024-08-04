import numpy as np
import pickle
import gym
import cv2

# Hyperparameters
H = 200  # Number of hidden layer neurons
batch_size = 10  # Episodes per parameter update
learning_rate = 1e-4
gamma = 0.99  # Discount factor for reward
decay_rate = 0.99  # RMSProp decay factor
resume = False  # Resume from previous checkpoint?
render = False  # Render the game

# Model Initialization
D = 80 * 80  # Input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # Xavier initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # Update buffers
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # RMSProp cache

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # Sigmoid function

def preprocess_frame(I):
    """ Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # Crop
    I = I[::2, ::2, 0]  # Downsample by factor of 2
    I[I == 144] = 0  # Erase background (background type 1)
    I[I == 109] = 0  # Erase background (background type 2)
    I[I != 0] = 1  # Everything else (paddles, ball) set to 1
    return I.astype(np.float).ravel()

def discount_rewards(r):
    """ Compute discounted rewards """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # Reset sum at game boundary
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # Return probability of action 2, and hidden state

def policy_backward(eph, epdlogp, epx):
    """ Backward pass """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # Backprop through ReLU
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}

env = gym.make("AirRaid-v0")
observation = env.reset()
prev_x = None  # Used in computing the difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    if render: env.render()

    # Preprocess the observation, set input to network to be difference image
    cur_x = preprocess_frame(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # Forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3  # Roll the dice!

    # Record various intermediates (needed later for backprop)
    xs.append(x)  # Observation
    hs.append(h)  # Hidden state
    y = 1 if action == 2 else 0  # A "fake label"
    dlogps.append(y - aprob)  # Grad that encourages the action taken

    # Step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)  # Record reward

    if done:  # An episode finished
        episode_number += 1

        # Stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # Reset array memory

        # Compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # Standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # Modulate the gradient with advantage
        grad = policy_backward(eph, epdlogp, epx)
        for k in model: grad_buffer[k] += grad[k]  # Accumulate grad over batch

        # Perform RMSProp parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # Gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # Reset batch gradient buffer

        # Boring bookkeeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print(f'Resetting env. Episode reward total was {reward_sum}. Running mean: {running_reward}')
        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()  # Reset env
        prev_x = None

    if reward != 0:  # AirRaid has either +1 or -1 reward exactly when game ends.
        print(f'ep {episode_number}: game finished, reward: {reward}') + ('' if reward == -1 else ' !!!!!!!!')

