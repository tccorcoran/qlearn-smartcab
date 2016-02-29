import random
import math
from pdb import set_trace
from collections import defaultdict
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.q_table = defaultdict(float)                                                                                                                                         
        self.A_previous = None
        self.S_previous = None
        self.max_q = defaultdict(tuple)
        self.alpha_table = defaultdict(tuple)
        self.T = 0
        self.gamma = 0.8
        self.epsilon = 0.75
        # TODO: Initialize any additional variables here
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.A_previous = None
        self.S_previous = None
        self.next_waypoint = None
        
    def updateAlpha(self,t):
        """ Alpha as a function of how many times
            we visit a particular state, tending to 0 in the limit
        """
        t += 1
        alpha = 1.0/(t)**2
        return (alpha,t)
    

    def update(self, t):
        """
        via: https://discussions.udacity.com/t/next-state-action-pair/44902/11
        Q-learning method 1:
        1) Sense the environment (see what changes naturally occur in the environment)
        2) Take an action - get a reward
        3) Sense the environment (see what changes the action has on the environment)
        4) Update the Q-table
        5) Repeat
        """
        #set_trace()
        #1) Sense the environment
        self.next_waypoint = self.planner.next_waypoint()   
        self.S_previous = (('light', self.env.sense(self)['light']),('waypoint',self.next_waypoint))
        
        
        valid_actions = [None, 'forward', 'left', 'right']
        rand_act = random.choice(valid_actions)
        # Select action according to epsilon greedy algorithm
        if len(self.max_q[self.S_previous]) > 0 and random.random() < self.epsilon:
            # pick argmax{a in A} Q(s,a)
            action = self.max_q[self.S_previous][1]
        else:
            # Pick random direction with P==(1-epsilon)
            action = rand_act
        # 2) Take an action - get a reward
        self.reward_previous = self.env.act(self, action)

        # 3) Sense the environment
        self.next_waypoint = self.planner.next_waypoint()   # from route planner, also displayed by simulator
        deadline = self.env.get_deadline(self)
        self.state = (('light', self.env.sense(self)['light']),('waypoint',self.next_waypoint))
            
        
        # Find the max q value of this state through all actions
        if len(self.max_q[self.state]) == 0:
            MAX_Q = 0
        else:
            MAX_Q = self.max_q[self.state][0]

        # Update the learning rate--a fn of number of visits to s
        if len( self.alpha_table[self.S_previous]) == 0:
            self.alpha_table[self.S_previous] = self.updateAlpha(1)
        else:
            self.alpha_table[self.S_previous] = self.updateAlpha(self.alpha_table[self.S_previous][1])
        alpha = self.alpha_table[self.S_previous][0]
        
        # 4) Update the Q-table
        q_sa = self.q_table[(self.S_previous,self.A_previous)] # Q(s,a) 
        # Q(s,a) <- Q(s,a) + alpha[r' + gamma*max{a'} Q(s',a') - Q(s,a)]
        self.q_table[(self.S_previous,self.A_previous)] = q_sa + alpha*(self.reward_previous + self.gamma*MAX_Q - q_sa)
        
        # Store the biggest q value for this state (along with the action that caused the large q, so we can pick that action again in step 2)
        if MAX_Q  <  self.q_table[(self.S_previous,self.A_previous)]:
            self.max_q[self.state]  =  (self.q_table[(self.S_previous,self.A_previous)], self.A_previous)
        
        
        print "State: ", self.S_previous, action
        print "Reward: ", self.reward_previous
        print "Deadline: ", deadline
        print "Alpha: ", alpha
        print "Q-value learned: ", self.q_table[(self.S_previous,self.A_previous)]
        print '-'*10
        #set_trace()             

        self.A_previous = action
        #print self.q_table
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, self.reward)  # [debug]



def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
