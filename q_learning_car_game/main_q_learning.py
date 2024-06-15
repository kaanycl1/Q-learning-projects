# Kaan Yücel - 150210318
import random
from collections import namedtuple
import time

import numpy as np
import csv
from game_q_learning import CarGameQL
from macros import *

# Are we rendering or not
RENDER_TRAIN = False

"""
    Parameters related to training process and Q-Learning
"""

# Number of episodes to train
NUM_EPISODES = 500

# epsilon parameter of e-greedy policy
EPSILON = 0.2

# learning rate parameter of q learning
LEARNING_RATE = 0.3

# discount rate parameter of q learning
DISCOUNT_RATE = 0.9

# The step limit per episode, since we do not want infinite loops inside episodes
MAX_STEPS_PER_EPISODE = 200

"""
    Parameters end
"""

# Here, we are creating the environment with our predefined observation space
env = CarGameQL(render=RENDER_TRAIN)

# Observation and action space
obs_space = env.observation_space
number_of_states = env.observation_space.shape[0]

action_space = env.action_space
number_of_actions = env.action_space.n
print("The observation space: {}".format(obs_space))
# Output: The observation space: Box(n,)
print("The action space: {}".format(action_space))
# Output: The action space: Discrete(m)

q_table = {}
actions = ["L", "R", "P"]


def choose_action_greedy(state, q_table):
    # if it is first move
    if state not in q_table.keys():
        return "L"

    # selecting the highest q score index
    action_index = np.argmax(q_table[state])

    # selecting the highest q score value
    action = actions[action_index]

    return action


def choose_action_e_greedy(state, q_table):
    # if it is first move
    if state not in q_table.keys():
        return "L"

    # random number for explore or exploit
    random_num = random.uniform(0, 1)
    if random_num > EPSILON:
        # exploit

        # selecting the highest q score index
        action_index = np.argmax(q_table[state])

        # selecting the highest q score value
        action = actions[action_index]
    else:
        # explore

        # take random action
        random_action = random.randint(0, 2)
        action = actions[random_action]

    return action


def save_q_table_csv(q_table, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow(["State", "Action", "Q-Value"])
        # Write data rows
        for state, q_value in q_table.items():
            for action_index in range(0, 3):
                actions = ["L", "R", "P"]
                writer.writerow([state, actions[action_index], q_value[action_index]])


def main():
    #  "Loop for each episode:"
    for e in range(NUM_EPISODES):
        #  "Initialize S"
        s0 = env.reset()

        #  "Loop for each step of episode:"
        episode_steps = 0
        while (episode_steps < MAX_STEPS_PER_EPISODE):
            episode_steps += 1
            #
            #  "Choose A from S using policy derived from Q (e.g., e-greedy)"
            #
            # policy chosen
            # action = choose_action_e_greedy(s0, q_table)
            action = choose_action_greedy(s0,q_table)
            #  "Take action A, observe R, S'"
            s1, reward, done, info = env.step(action)

            # assigning zero to q_table if that state never been encountered before
            if s1 not in q_table:
                q_table[s1] = np.zeros(3)
            if s0 not in q_table:
                q_table[s0] = np.zeros(3)

            # q_table is dictionary which states as keys each state has numpy array of length 3.First index of the
            # numpy array is action "L" second is "R" and last is "P".Therefore q_table is like this:
            # q_table={state:[-26,1,5]} For instance,-26 is the q value for the state-action pair (state,"L")

            # taking the index of an action
            if action == "L":
                action_index = 0
            elif action == "R":
                action_index = 1
            else:
                action_index = 2
            # max q value index in the state "s1"
            max_a = np.argmax(q_table[s1])

            #  "Q(S,A) <-- Q(S,A) + alpha*[R + gamma* maxa(Q(S', a)) - Q(S, A)]"

            q_table[s0][action_index] = q_table[s0][action_index] + LEARNING_RATE * (
                    reward + DISCOUNT_RATE * (q_table[s1][max_a])
                    - q_table[s0][action_index])
            #  "S <-- S'"
            s0 = s1

            # until S is terminal
            if (done):
                break

        #  print number of episodes so far
        if (e % 100 == 0):
            print("episode {} completed".format(e))

    # Call this function after training to save your Q-table
    save_q_table_csv(q_table, 'q_table.csv')

    #  test our trained agent
    test_agent(q_table)


def test_agent(q_table):
    print("Initializing test environment:")
    test_env = CarGameQL(render=True, human_player=False)
    state = env.reset()
    steps = 0
    # while (steps < 200):
    while (True):
        action = choose_action_greedy(state, q_table)
        print("chosen action:", action)
        next_state, reward, done, info = test_env.step(convert_direction_to_action(action))
        print("state:", state, " , next_state:", next_state)
        test_env.render()
        if done:
            break
        else:
            state = next_state
        steps += 1
        print("test current step:", steps)

        time.sleep(0.1)


if __name__ == '__main__':
    main()
