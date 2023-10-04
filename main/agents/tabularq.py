import numpy as np
from itertools import product

class StateMapper:
    def __init__(self,
                 num_bins=15,
                 lower=1.47293,
                 upper=1.92498,
                 xi=0.1):
        self.bins = np.round(np.linspace(lower - xi*(upper-lower),
                                         upper + xi*(upper-lower),
                                         num_bins), 15)
        
        self.bin_idx = [str(k).zfill(2) for k in np.arange(0,num_bins)]
        
        self.mapping_dict = dict(zip(self.bins, self.bin_idx))
        
        self.all_state_dict = self._all_states_mapper()
               
    def _all_states_mapper(self):
        all_combs = list(product(self.bin_idx, self.bin_idx))
        joined_combs = [''.join(tu) for tu in all_combs]
        
        return dict(zip(joined_combs, np.arange(0,len(joined_combs))))
    
    def __call__(self, p1, p2):
        mapped_bin_1 = self.mapping_dict[p1]
        mapped_bin_2 = self.mapping_dict[p2]
        return self.all_state_dict[mapped_bin_1+mapped_bin_2]
    

class QLearningAgentBase:
    def __init__(self,
                 alpha,
                 epsilon,
                 gamma=0.99,
                 n_actions=15,
                 n_states=225,
                 n_steps=1):

        self.n_actions = n_actions
        self.n_states = n_states

        self._qvalues = np.random.rand(self.n_states, self.n_actions)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_steps = n_steps

    def set_qmatrix(self, new_matrix):
        self._qvalues = new_matrix

    def get_qvalue(self, state, action):
        """
        Returns the Q-value for the given state action
        Args:
            state (integer): Index representation of a state
            action (integer): Index representation of an action
        Returns:
            float: Q-value for the state-action combination
        """
        return self._qvalues[state, action]

    def set_qvalue(self, state, action, value):
        """Sets the Qvalue for [state,action] to the given value
        Args:
            state (integer): Index representation of a state
            action (integer): Index representation of an action
            value (float): Q-value that is being assigned
        """
        self._qvalues[state, action] = value

    def get_value(self, state):
        """
        Compute the agents estimate of V(s) using current q-values.
        Args:
            state (integer): Index representation of a state
        Returns:
            float: Value of the state
        """
        value = np.max(
            self._qvalues[
                state,
            ]
        )

        return value

    def get_qmatrix(self):
        """
        Returns the qmatrix of the agent
        Returns:
            array (float): Full Q-Matrix
        """

        return self._qvalues

    def update(self, state, action, reward, next_state):
        """
        Update Q-Value:
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        Args:
            state (integer): Index representation of the current state (Row of the Q-matrix)
            action (integer): Index representation of the picked action (Column of the Q-matrix)
            reward (float): Reward for picking from picking the action in the given state
            next_state (integer): Index representation of the next state (Column of the Q-matrix)
        """
        # Calculate the updated Q-value
        c_q_value = (1 - self.alpha) * self.get_qvalue(state, action) + self.alpha * (
            reward + (self.gamma**self.n_steps) * self.get_value(next_state)
        )

        # Update the Q-values for the next iteration
        self.set_qvalue(state, action, c_q_value)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        Args:
            state (integer): Index representation of the current state (Row of the Q-matrix)
        Returns:
            integer: Index representation of the best action (Column of the Q-matrix)
                     for the given state (Row of the Q-matrix)
        """

        # Pick the Action (Row of the Q-matrix) with the highest q-value
        best_action = np.argmax(self._qvalues[state, :])
        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we take a random action.
        Returns both, the chosen action (with exploration) and the best action (argmax).
        If the chosen action is the same as the best action, both returns will be
        the same.
        Args:
            state (integer): Integer representation of the current state (Row of the Q-matrix)
        Returns:
            tuple: chosen_action, best_action
                   chosen_action (integer): Index representation of the acutally picked action
                                            (Column of the Q-matrix)
                   best_action (integer): Index representation of the current best action
                                          (Column of the Q-matrix) in the given state.
        """
        # agent parameters:
        epsilon = self.epsilon
        e_threshold = np.random.random()

        # Get the best action.
        best_action = self.get_best_action(state)

        if e_threshold < epsilon:
            # In the numpy.random module randint() is exclusive for the upper
            # bound and inclusive for the lower bound -> Actions are array
            # indices for us.
            chosen_action = np.random.randint(0, self.n_actions)
        else:
            chosen_action = best_action
        return chosen_action, best_action
    

    