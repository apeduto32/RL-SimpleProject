# Import package
import sys
import numpy as np

# Implementation of the class RLInteraction. It'll be based on the assumption that
# an agent is interacting with an environment to solve an episodic task
class RLInteraction(object):
    # Constructor
    def __init__(self):
        """
        Constructor used for initialize an instance's attributes
        """
        # Initialize the number of episodes
        self.num_episodes = 0
        # Initialize the list of sum of reward for all future episodes
        self.sum_rewards_episodes = []
        # Initialize the list of time steps for all future episodes
        self.time_steps_episodes = []
        # initialize the time steps for one episode
        self.time_steps_episode = 0
        # Initialize the total number of time steps
        self.time_steps = 0

    
    # Method used to start an interaction between agent and environment    
    def __start(self, agent, env):
        """
        Method used for the first interaction between agent and environment.
        Params
        ======
            - agent (obj) --> Instance of one of the classes in MazeAgents
            - env (obj) --> Instance of the class MazeEnvironment
        """
        # Get the starting state of agent by env
        starting_state = env.reset()
        # Update window of environment
        env.render()
        # get the first action of agent
        first_action = agent.start(starting_state)
    
        # save start_state and first_action in attributes 
        # last_state and last_action
        self.last_state = starting_state
        self.last_action = first_action
    
        # Set number of steps for episode equal to 1
        self.time_steps_episode = 1
        # Increment the number of time steps by 1
        self.time_steps += 1
    
    # Method used to execute a step of interaction between agent and environment
    def __step(self, agent, env):
        """
        Method used to execute one step involving an agent and the environment.
        Params
        ======
            - agent (obj) --> Instance of one of the classes in MazeAgents
            - env (obj) --> Instance of the class MazeEnvironment
        Returns
        =======
            - observation (tuple) --> Tuple containing information about 
            reward and termination
        """
        # Get observation from the environment env
        (reward, state, done) = env.step(self.last_action)
        # update window
        env.render()
    
        # Initialize action to None
        action = None
        # Check if the signal of termination is True
        if done == True:
            # Execute the last update of q-values and policy inside of 
            # agent by its method agent_end
            agent.end(reward)
        # otherwise
        else:
            # get the next action and update the q-values and policy inside of 
            # agent by its method agent_step
            action = agent.step(reward, state)
        

        # Update the attributes last_state and last_action
        self.last_state = state
        self.last_action = action
    
        # Update the number of time steps for episode
        self.time_steps_episode += 1
        # Update the total number of time steps
        self.time_steps += 1

        # return the observation done
        return (reward, done)
    
    # Method used to generate one episode
    def __run_episode(self, agent, env, max_steps_episode = np.inf):
        """
        Method used to execute one episode.
        Params
        ======
            - agent (obj) --> Instance of one of the classes in MazeAgents
            - env (obj) --> Instance of the class MazeEnvironment
            - max_steps_episode (int) --> maximum number of steps for one episode 
        Output:
            - completed (boolean) --> boolean to indicate if an episode is 
            complete (True) or not (False)
        """
        # Check that the input value is positive
        assert max_steps_episode > 0,\
        'A value of maximum number of steps for episode not valid'
        
        
        # Starting of an episode by using the method start
        self.__start(agent, env)

        # Initialize the sum of rewards
        sum_rewards = 0.
        # Initialize boolean complete to false
        completed = False

        # Until the goal has not been reached and 
        # number of steps for episode is less than the maximum value
        while(completed==False and self.time_steps_episode < max_steps_episode):
            # execute the method step
            (reward, done) = self.__step(agent, env)
            # update completed
            completed = done
            # update the sum of rewards
            sum_rewards += reward
    
        # update the number of episodes
        self.num_episodes += 1
        # append the sum of rewards to the attribute sum_rewards_for_episode
        self.sum_rewards_episodes.append(sum_rewards)
        # append num_steps_for_episode in list time_steps_for_episode
        self.time_steps_episodes.append(self.time_steps_episode)

        # Return a boolean to say if the episode is complete 
        # (goal has been reached) or not
        return completed
    
    def run_experiment(self, agent, env, num_episodes, max_steps_episode=np.inf):
        """
        Method used to execute an experiment for aget interacting with 
        environment.
        Params
        ======
            - agent (obj) --> Instance of one of the classes in MazeAgents
            - env (obj) --> Instance of the class MazeEnvironment
            - num_episodes (int) --> Integer for the number of episodes to execute
            - max_steps_episode (int) --> Max number of time steps for one episode
        """
        # Initialize the following variables
        num_episodes_completed = 0
        best_return = -np.inf
        # for each episode
        for i in range(1,num_episodes+1):
            # execute its
            completed = self.__run_episode(agent, env, max_steps_episode)
            # update num_episodes_completed
            num_episodes_completed += (completed == True)
            # update best return
            best_return = max(best_return, self.sum_rewards_episodes[-1])
            # print episode
            print("\rEpisode {}/{} || best return: {}".format(i, num_episodes, best_return), end="")
            sys.stdout.flush()

        
        # Print the number of episodes completed
        print("\nEpisodes {}/{} completed with success.".format(num_episodes_completed, num_episodes), end="")
    
    
    # Method to cleaup agent, env and other attributes
    def cleanup(self):
        """
        Function used to cleanup an object's attributes.
        """
        self.last_state = None
        self.last_action = None

        # Initialize the number of episodes
        self.num_episodes = 0
        # Initialize the list of sum of reward for all future episodes
        self.sum_rewards_episodes.clear()
        # Initialize the list of time steps for all future episodes
        self.time_steps_episodes.clear()
        # initialize the time steps for one episode
        self.time_steps_episode = 0
        # Initialize the total number of time steps
        self.time_steps = 0

# Test for the class RLInteraction
def main():
    from MazeEnvironment import MazeEnvironment
    from MazeAgents import MazeSarsaAgent, MazeExpSarsaAgent, MazeQAgent
    import matplotlib.pyplot as plt
    print('Test for RL Interaction')
    # Define the environment
    height, width = 6, 6
    num_states = height*width
    env_info = {'height':height, 'width':width, 'start_loc':(0,0),'goal_loc':(5,5), \
        'obstacles':[(2,2),(3,3)], 'time_sleep':.0001}
    env = MazeEnvironment(env_info)
    # Define the agent
    num_actions = 9
    agent_info = {'num_states':num_states, 'num_actions':num_actions, \
        'epsilon':.1, 'alpha':.4, 'gamma':0.9}
    mSarsaAgent = MazeSarsaAgent(agent_info)
    #mExpSarsaAgent = MazeExpSarsaAgent(agent_info)
    #mQAgent = MazeQAgent(agent_info)
    # Define the number of episodes
    num_episodes = 300
    # Create instance of RLInteraction
    rl = RLInteraction()
    # Run experiment 
    rl.run_experiment(mSarsaAgent, env, num_episodes)
    #rl.run_experiment(mDynaQAgent, env, num_episodes)
    #rl.run_experiment(mQAgent, env, num_episodes)
    # plot results
    x = np.arange(1, num_episodes+1)
    y = rl.sum_rewards_episodes
    plt.plot(x, y)
    plt.xlabel('Episodes')
    plt.ylabel('Sum rewards')
    plt.show()

if __name__ == '__main__':
    main()