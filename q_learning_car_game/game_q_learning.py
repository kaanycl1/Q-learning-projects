# Kaan Yücel - 150210318
from game import *


class CarGameQL(CarGame):
    def __init__(self, render=True, human_player=False):
        super(CarGameQL, self).__init__(render, human_player)

        """
        DEFINE YOUR OBSERVATION SPACE DIMENSIONS HERE FOR EACH MODE.
        JUST CHANGING THE "obs_space_dim" VARIABLE SHOULD BE ENOUGH
        
            Try making your returned state from get_state function
        a 1-D array if its not, it will make things simpler for you
        
            For the first Q-Learning part, you must use a more compact
        game state than raw game array
        """
        obs_space_dim = 16
        self.observation_space = spaces.Box(0, obs_space_dim, shape=(obs_space_dim,))

    def get_state(self):
        """
        Define your state representation here
        
        self.game_state gives you original [6][5] game grid array
        """
        # State is represented as player's position and then the positions that possibly belong to blue cars excluding
        # last row because last row does not affect the outcome
        # For example ((5, 1),'e', 'b', 'e', 'e', 'e', 'e', 'e', 'e', 'b', 'e', 'e', 'e', 'e' ,'e' , 'b')
        #             player,[0,0][0,2][0,4][1,0]...................................................[4,4]
        # this way we get only the necessary positions that have possibly belongs to blue cars for simpler
        # and smaller state representation

        state = [(self.player_row, self.player_column)]
        # enemy positions
        for k in range(0, 5):  # taking the rows up to 5th
            for i in range(0, 5, 2):  # taking the columns [0,2,4]
                state.append(self.game_state[k][i])
        return tuple(state)

    def get_reward(self):
        """
        Define your reward calculations here
        """
        # Rewards are given as -99 for crashing and 1 for not crashing.
        if self.IsCrashed():
            self.reward = -99
        else:
            self.reward = 1

        self.total_reward += self.reward
        return self.reward
