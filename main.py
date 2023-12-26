import random
import gym
import gym_maze
import numpy as np

# Create an environment
env = gym.make("maze-random-10x10-plus-v0")
observation = env.reset()

qtable = np.zeros((100, 4))
learning_rate = 0.8
max_steps = 99
gamma = 0.95
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01


# Define the maximum number of iterations
NUM_EPISODES = 1000

for episode in range(NUM_EPISODES):
    env.render()
    state = env.reset()
    state = int(state[0] * 10 + state[1])
    step = 0
    done = False
    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0, 1)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        else:
            action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        new_state = int(new_state[0] * 10 + new_state[1])
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        state = new_state
        if done == True: 
            break
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 


    # TODO: Implement the agent policy here
    # Note: .sample() is used to sample random action from the environment's action space

    # Choose an action (Replace this random action with your agent's policy)
    #action = env.action_space.sample()

    # Perform the action and receive feedback from the environment
    next_state, reward, done, truncated = env.step(action)

    if done or truncated:
        observation = env.reset()

# Close the environment
env.close()
