# Import packages
import numpy as np
from BaseAgent import BaseAgent

# Implementation of class that defines the attributes and methods of an agent in a maze.
# The learning algorithm will be defined in a child class
class MazeAgent(BaseAgent):
    # Constructor
    def __init__(self, agent_info={}):
        """
        Define the attributes of Agent in a tabular task.
        Params:
            - agent_info (dict) --> Dictionary containing information to define a Maze Agent
        """
        # Initialize a random-number generator with random seed
        self.random_generator = np.random.RandomState(seed=agent_info.get('seed',None))

        # Initialize number of states
        self.num_states = agent_info.get('num_states')
        # Initialize number of actions
        self.num_actions = agent_info.get('num_actions')

        # We'll base on the assumption to be in a tabular task with finite number of states and actions,
        # so the action-value function and the policy will have the shape (#num_states, #num_action)
        # Initialize the action-values function to a zeros array
        self.action_values = np.zeros(shape=(self.num_states, self.num_actions))
        # Initialize the parameter epsilon to have an epsilon-greedy policy
        self.epsilon = agent_info.get('epsilon',.1)
        # Initialize the parameter alpha
        self.gamma = agent_info.get('gamma',1.)
        # Initialize the parameter gamma
        self.alpha = agent_info.get('alpha',.5)


    def _argmax(self, state):
        """
        Method used to get the action corresponding to the max value of state-action function for an
        input state.
        Params:
            - state (int) --> Integer representing a state
        Returns:
            - action (int) --> Integer representing an action
        """
        # Extract action-values corrensponding to the input state
        q_state_values = self.action_values[state,:]
        # get the max value of q_state_values
        q_max = np.amax(q_state_values)

        # get the list of actions corresponding to q_max
        list_actions = []
        for i in range(self.num_actions):
            if q_state_values[i] == q_max:
                list_actions.append(i)

        # return action corresponding to q_max
        action = None
        if len(list_actions) == 1:
            action = list_actions[0]
        elif len(list_actions) > 1:
            action = self.random_generator.choice(list_actions)
        else:
            raise Exception('No action found for q_max = {}'.format(q_max))

        return action

    def _choose_action(self, state):
        """
        Method used to choose an action given an input state
        Params:
            - state --> Integer representing a state
        Returns:
            - action --> Integer representing the choosen action
        """
        # Get a random numeric value inside of the range (0,1]
        p = self.random_generator.uniform()
        # Choose an action by using an epsilon-greedy policy
        action = None
        if p < self.epsilon:
            action = self.random_generator.choice(self.num_actions)
        else:
            action = self._argmax(state)

        # return the choosen action
        return action

    def start(self, state):
        """
        Method used for the first action of an agent
        Params:
            - state (int) --> Number representing a state
        Returns:
            - action (int) --> Number representing the choosen action
        """
        # Choose the first action by using the policy
        action = self._choose_action(state)
        # Initialize last_state
        self.last_state = state
        # Initialize last_action
        self.last_action = action

        # return the action
        return action

    def _apply_algorithm(self, reward, next_state, next_action, done):
        """
        Method used to teach an agent to interact with an environment with an algorithm
        to specify.
        """
        pass # No algorithm

    def step(self, reward, next_state):
        """
        Method used to update:
            - action-values function of the previous pair state-action
            - policy of the previous state
            - last state and last action
        Params:
            - reward (float) --> Float value received by environment
            - next_state (int) --> Integer value corresponding to the next state
        Returns:
            - action (int) --> Integer value representing the choosen action
        """
        # Choose the action, given the next state from environment
        next_action = self._choose_action(next_state)

        # Update action-value function by an algorithm specified in the method apply_algorithm
        self._apply_algorithm(reward, next_state, next_action, done=False)

        # Update the attributes last_state and last_action
        self.last_state = next_state
        self.last_action = next_action

        # return the action selected for the next state
        return next_action

    def end(self, reward):
        """
        Method used when the terminal state has been reached to update:
            - action-values function of the previous pair state-action
            - policy of the previous state
        Params:
            - reward --> Float value received by environment
        Returns: None
        """
        # Update action-value function by an algorithm specified in the method apply_algorithm
        self._apply_algorithm(reward, next_state=None, next_action=None, done=True)

    def cleanup(self):
        """
        Method used to clean memory of agent
        """
        """
        Method used to clean memory of agent
        """
        # Clean lasst_state, last_action
        self.last_state = None
        self.last_action = None
        self.action_values = np.zeros(shape=(self.num_states, self.num_actions))

# --------------------------Maze Sarsa-Agent-------------------------------------------------
# Implementation of child class based on the Sarsa Learning algorithm
class MazeSarsaAgent(MazeAgent):
    # Constructor
    def __init__(self, agent_info):
        super(MazeSarsaAgent, self).__init__(agent_info)


    # Overriding of the method apply_algorithm
    def _apply_algorithm(self, reward, next_state, next_action, done):
        """
        Method used to implement the Q-Learning algorithm.
        Params:
            - reward (float) --> Float value received by environment
            - next_state (int) --> Integer value representing the next state
            - next_action (int) --> Integer value representing the next action
            - done (boolean) --> Boolean true if the goal has benn reached otherwise false
        """
        # Define old_estimate
        old_estimate = self.action_values[self.last_state, self.last_action]
        # Intialize target
        target = reward
        # Check if done is false
        if done == False:
            # Update target
            target += self.gamma*self.action_values[next_state, next_action]
        # update action-value
        delta = target - old_estimate
        self.action_values[self.last_state, self.last_action] += self.alpha*delta

# --------------------------Maze Expected Sarsa-Agent-------------------------------------------------
# Implementation of child class based on the Expected Sarsa Learning algorithm
class MazeExpSarsaAgent(MazeAgent):
    # Constructor
    def __init__(self, agent_info):
        super(MazeExpSarsaAgent, self).__init__(agent_info)
        # initialize policy
        self.policy = np.ones(shape=(self.num_states, self.num_actions))/self.num_actions

    # Method used to update policy
    def _update_policy(self, state):
        """
        Method used to update policy
        """
        self.policy[state, :] = np.ones(self.num_actions)*(self.epsilon/self.num_actions)
        index = np.argmax(self.action_values[state,:])
        self.policy[state, index] += 1. - self.epsilon

    # Overriding of the method apply_algorithm
    def _apply_algorithm(self, reward, next_state, next_action, done):
        """
        Method used to implement the Q-Learning algorithm.
        Params:
            - reward (float) --> Float value received by environment
            - next_state (int) --> Integer value representing the next state
            - next_action (int) --> Integer value representing the next action
            - done (boolean) --> Boolean true if the goal has benn reached otherwise false
        """
        # Define old_estimate
        old_estimate = self.action_values[self.last_state, self.last_action]
        # Intialize target
        target = reward
        # Check if done is false
        if done == False:
            # Update target
            target += self.gamma*np.sum(self.policy[next_state,:]*self.action_values[next_state,:])
        # update action-value
        delta = target - old_estimate
        self.action_values[self.last_state, self.last_action] += self.alpha*delta
        # update policy
        self._update_policy(self.last_state)
    

# --------------------------Maze Q-Agent-------------------------------------------------
# Implementation of child class based on the Q-Learning algorithm
class MazeQAgent(MazeAgent):
    #Constructor
    def __init__(self, agent_info):
        super(MazeQAgent, self).__init__(agent_info)
    # Overriding of the method apply_algorithm
    def _apply_algorithm(self, reward, next_state, next_action, done):
        """
        Method used to implement the Q-Learning algorithm.
        Params:
            - reward (float) --> Float value received by environment
            - next_state (int) --> Integer value representing the next state
            - next_action (int) --> Integer value representing the next action
            - done (boolean) --> Boolean true if the goal has benn reached otherwise false
        """
        # Define old_estimate
        old_estimate = self.action_values[self.last_state, self.last_action]
        # Intialize target
        target = reward
        # Check if done is false
        if done == False:
            # Update target
            target += self.gamma*np.amax(self.action_values[next_state,:])
        # update action-value
        delta = target - old_estimate
        self.action_values[self.last_state, self.last_action] += self.alpha*delta

    

#-----------------------------Maze Dyna Q-Agent-------------------------------------------------
# Implementation of child class of Q-Agent with direct and indirect reinforcement learning (RL)
class MazeDynaQAgent(MazeQAgent):
    # Constructor
    def __init__(self,agent_info={}):
        """
        Method used to define a Maze Dyna Q-Agent.
        Params:
            - agent_info (dict) --> Dictionary containing information to define a Maze Dyna Q-Agent
        """
        # Recall mother class' constructor
        super(MazeDynaQAgent,self).__init__(agent_info)
        # initialize the sample-model
        self.sample_model = {}
        # initialize the number of planning steps for indirect RL
        self.planning_steps = agent_info.get('planning_steps',20)
        # Initialize random generator for planning (indirect RL)
        self.planning_random_generator = np.random.RandomState(seed=agent_info.get('seed',None))


    def _update_model(self, past_state, past_action, state, reward, done):
        """
        Method used to update the sample-model by experience with the environment.
        Params
        ======
            - past_state (int) --> Integer value representing the last state
            - past_action (int) --> Integer value representing the last action
            - state (int) --> Integer value representing a state observed by the environment
            - reward (float) --> Float value received by environment
            - done (bool) --> Boolean True if the goal has been reached otherwise False
        """
        # Check if the key number of past_state is not already present into the dictionary sample_model
        if past_state not in list(self.sample_model.keys()):
            # add it in the sample-model
            self.sample_model[past_state] = {past_action:(state, reward, done)}
            # update sample-model for other actions
            for action in range(self.num_actions):
                if action != past_action:
                    self.sample_model[past_state][action] = (past_state, 0, False)
        # otherwise
        else:
            # Update the sample_model for past_state and past_action
            self.sample_model[past_state][past_action] = (state, reward, done)


    def _planning(self):
        """
        Method used to update the action-values function (policy) by a simulated experience generated from
        the sample-model.
        """
        # check if planning_steps is more than zero
        if self.planning_steps > 0:
            # For each step of planning
            for _ in range(self.planning_steps):
                # Choose a random past pair state-action from the dictionary sample-model
                past_state = self.planning_random_generator.choice(list(self.sample_model.keys()))
                past_action = self.planning_random_generator.choice(list(self.sample_model[past_state].keys()))
                # get the corresponding past pair state-reward
                (state, reward, done) = self.sample_model[past_state][past_action]
                # apply the algorithm
                if done:
                    self._apply_algorithm(reward=reward, next_state=None,next_action=None,done=True)
                else:
                    self._apply_algorithm(reward=reward, next_state=state,next_action=None,done=False)
        # otherwise
        else:
            pass

    # Overriding of the method step
    def step(self, reward, state):
        """
        Method used to apply direct and indirect Reinforcement Learning (RL)
        Params:
            - state (int) --> Integer value representing a state
            - reward (float) --> Float value received by environment
        Returns:
            - action (int) --> Integer value representing the action choosen by agent
        """
        ## Direct RL
        self._apply_algorithm(reward=reward, next_state=state, next_action=None, done=False)
        ## Update the sample-model of environment
        self._update_model(past_state=self.last_state, past_action=self.last_action, \
            state=state, reward=reward, done=False)
        ## Planning (Indirect RL)
        self._planning()

        # Choose the next action
        action = self._choose_action(state)

        # Update last_state and last_action
        self.last_state = state
        self.last_action = action

        # Return the choosen action
        return action

    # Overriding of the method end
    def end(self, reward):
        """
        Method used to execute the last step.
        Input:
            - reward (float) --> Float Value representing the last reward of episode received from environment.
        """
        ## Direct RL
        self._apply_algorithm(reward=reward, next_state=None, next_action=None, done=True)
        ## Update the sample-model of environment
        self._update_model(past_state=self.last_state, past_action=self.last_action,\
            state=None, reward=reward, done=True)
        ## Planning (Indirect RL)
        self._planning()

    # Overriding of the method cleanup
    def cleanup(self):
        """
        Method used to clear memory of angent.
        """
        # Initialize state-action values
        self.action_values = np.zeros((self.num_states,self.num_actions))
        # clean last_state and last_action
        self.last_state = None
        self.last_action = None
        # clean the sample-model
        self.sample_model.clear()

# Test MazeAgent
def main():
    agent_info = {'num_states':9, 'num_actions':9}
    # Test of MazeAgent
    print('Test of MazeAgent')
    mAgent = MazeAgent(agent_info)
    start_state = 1
    print('start state:', start_state)
    first_action = mAgent.start(start_state)
    print('first action:', first_action)
    reward = -1.
    next_state = 2
    print('reward:', reward, 'next state:', next_state)
    next_action = mAgent.step(reward, next_state)
    print('next action:', next_action)

if __name__ == '__main__':
    main()
    
