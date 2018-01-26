import numpy as np
import pandas as pd
import time
import copy

class learner():
    def __init__(self,**args):
        # get some crucial parameters from the input gridworld
        self.grid = args['gridworld']
        
        # initialize q-learning params
        self.gamma = 1
        self.max_steps = 5*self.grid.width*self.grid.height
        self.exploit_param = 0.5
        self.action_method = 'exploit'
        self.training_episodes = self.grid.training_episodes
        self.validation_episodes = 50
        self.training_start_schedule = []
        self.validation_start_schedule = []
           
        # create start schedule for training / validation
        self.training_start_schedule = self.grid.training_start_schedule[:self.training_episodes]
        self.validation_start_schedule = self.grid.validation_start_schedule[:self.validation_episodes]
        
    ### Q-learning function - version 1 - take random actions ###
    def train(self,**args):
        # switches
        if "gamma" in args:
            self.gamma = args['gamma']
        if 'max_steps' in args:
            self.max_steps = args['max_steps']
        if 'action_method' in args:
            self.action_method = args['action_method']
        if 'exploit_param' in args:
            self.exploit = args['exploit_param']
            self.action_method = 'exploit'
        if 'training_episodes' in args:
            self.training_episodes = args['training_episodes']
            # return error if number of training episodes is too big
        if self.training_episodes > self.grid.training_episodes:
            print ('requesting too many training episodes, the maximum num = ' + str(self.grid.training_episodes))
            return        
        if 'validation_episodes' in args:
            self.validation_episodes = args['validation_episodes']
            # return error if number of training episodes is too big
        if self.validation_episodes > self.grid.validation_episodes:
            print ('requesting too many validation episodes, the maximum num = ' + str(self.grid.validation_episodes))
            return 
        
        # make local (non-deep) copies of globals for easier reading
        grid = self.grid
        gamma = self.gamma
        
        # containers for storing various output
        self.training_episodes_history = {}
        self.training_reward = []
        self.validation_reward = []
        self.time_per_episode = []
        self.Q_history = []
        Q = np.zeros((self.grid.width*self.grid.height,len(self.grid.action_choices)))
        self.Q_history.append(copy.deepcopy(Q))  # save current Q (for visualization of progress)

        ### start main Q-learning loop ###
        for n in range(self.training_episodes): 
            start = time.clock()
            
            # pick this episode's starting position
            grid.agent = self.training_start_schedule[n]

            # update Q matrix while loc != goal
            episode_history = []      # container for storing this episode's journey
            total_episode_reward = 0
            for step in range(self.max_steps):   
                # update episode history container
                episode_history.append(grid.agent)
                
                ### if you reach the goal end current episode immediately
                if grid.agent == grid.goal:
                    break
                
                # translate current agent location tuple into index
                s_k_1 = grid.state_tuple_to_index(grid.agent)
                    
                # get action
                a_k = grid.get_action(method = self.action_method,Q = Q,exploit_param = self.exploit_param)
                
                # move based on this action
                s_k = grid.get_movin(action = a_k)
               
                # get reward     
                r_k = grid.get_reward(state_index = s_k)          
                
                # update Q
                Q[s_k_1,a_k] = r_k + gamma*max(Q[s_k,:])
                    
                # update current location of agent 
                grid.agent = grid.state_index_to_tuple(state_index = s_k)
                
                # update training reward
                total_episode_reward+=r_k
            # print out update if verbose set to True
            if 'verbose' in args:
                if args['verbose'] == True:
                    if np.mod(n+1,100) == 0:
                        print ('training episode ' + str(n+1) +  ' of ' + str(self.training_episodes) + ' complete')
            
            ### store this episode's computation time and training reward history
            stop = time.clock()
            self.time_per_episode.append(stop - start)
            self.training_episodes_history[str(n)] = episode_history
            self.training_reward.append(total_episode_reward)
            self.Q_history.append(copy.deepcopy(Q))  # save current Q (for visualization of progress)

            ### store this episode's validation reward history
            if 'validate' in args:
                if args['validate'] == True:
                    reward = self.validate(Q)
                    self.validation_reward.append(reward)

        self.Q = Q  # make a global version
            
        print ('q-learning algorithm complete')
       
    ### run validation episodes ###
    def validate(self,Q):
        # make local (non-deep) copies of globals for easier reading
        grid = self.grid
        
        # run validation episodes
        total_reward = []

        # run over validation episodes
        for i in range(self.validation_episodes):  

            # get this episode's starting position
            grid.agent = self.validation_start_schedule[i]

            # reward container for this episode
            episode_reward = 0

            # run over steps in single episode
            for j in range(grid.max_steps):
                ### if you reach the goal end current episode immediately
                if grid.agent == grid.goal:
                    break
                
                # translate current agent location tuple into index
                s_k_1 = grid.state_tuple_to_index(grid.agent)
                    
                # get action
                a_k = grid.get_action(method = 'optimal',Q = Q)
                
                # move based on this action - if move takes you out of gridworld don't move and instead move randomly 
                s_k = grid.get_movin(action = a_k, illegal_move_response = 'random')
  
                # compute reward and save
                r_k = grid.get_reward(state_index = s_k)          
                episode_reward += r_k
    
                # update agent location
                grid.agent = grid.state_index_to_tuple(state_index = s_k)
                
            # after each episode append to total reward
            total_reward.append(episode_reward)

        # return total reward
        return np.median(total_reward)