from itertools import product
import numpy as np

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
        mapped_bin_2 = self.mapping_dict[p1]
        return self.all_state_dict[mapped_bin_1+mapped_bin_2]
        

class TabularQ:
    def __init__(self, 
                 state_mapper,
                 learning_rate,
                 n_states=225, 
                 n_actions=15,
                 gamma=0.95):
        self.state_mapper = state_mapper
        self.gamma = gamma
        self.lr = learning_rate
        self.n_states = n_states
        self.n_actions = n_actions
        self._set_table()
        self._set_action_dict()
        
    def _set_table(self):
        self.q_tab = np.random.rand(self.n_states, self.n_actions)
        
    def _set_action_dict(self):
        self.action_dict =  dict(zip(np.arange(0,self.n_actions),
                                     self.state_mapper.bins))
        
    def select_action(self, state, epsilon):
        if np.random.random() < epsilon:
            action =  np.random.randint(0, self.n_actions)
            return self.translate_action(action), action
        else:
            action = np.argmax(self.q_tab[state,:])
            return self.translate_action(action), action
    
    def translate_action(self, action):
        return self.action_dict[action]
        
    def update(self, state, action, reward, new_state):
        next_q_val = np.max(self.q_tab[new_state, :])
        target = reward + self.gamma*next_q_val # do not need to have game over as it is not episodic
        
        pol_grad = target - self.q_tab[state, action]
        
        self.q_tab[state, action] += self.lr*pol_grad
        
class SimpleEconEnvironment:
    def __init__(self, mu=0.25, a_vec=np.array([0,2,2])):
        self.mu = mu
        self.a_vec = a_vec
    
    def aggregate_demand(self, p1, p2):
        p_vec = np.array([0, p1, p2])
        all_terms = np.exp((self.a_vec - p_vec)/self.mu)
        return (all_terms / np.sum(all_terms))[1:]
    
    def __call__(self, p1, p2):
        return self.aggregate_demand(p1, p2)