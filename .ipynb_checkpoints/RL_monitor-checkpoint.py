from collections import deque
import sys
import math
import numpy as np

def interact(env, agent, num_episodes=20000):
    """ Monitor agent's performance.
    
    Params
    ======
    - env (obj): instance of the class MazeEnvironment
    - agent (obj): instance of class MazeAgent
    - num_episodes (int): number of episodes of agent-environment interaction

    Returns
    =======
    - returns: deque containing return of each episode
    """
    # Initialize a deque to save the average of rewards for each episode required
    returns = deque(maxlen=num_episodes)
    # for each episode
    for i_episode in range(1, num_episodes+1):
        # start it, getting the initial state of agent
        state = env.reset()
        # Update window of environment
        env.render()
        # get first action of agent
        action = agent.start(state)
        # initialize the sampled return
        sample_return = 0.
        while True:
            # apply the actio choosen by agent
            (reward, next_state, done) = env.step(action)
            # update window
            env.render()
            # update the sampled return
            sample_return += reward
            # update the state (s <- s') to next time step
            state = next_state
            # Check if the signal of termination is True
            if done:
                # Execute the last update of q-values and policy inside of agent
                agent.end(reward)
                # save sample_return in returns
                returns.append(sample_return)
                break
            # otherwise
            else:
                # get the next action and update the q-values and policy inside of agent
                action = agent.step(reward, next_state)
        # monitor progress
        print("\rEpisode {}/{} || return: {}".format(i_episode, num_episodes, sample_return), end="")
        sys.stdout.flush()
        if i_episode == num_episodes: print('\n')
    # close window
    env.close_window()
    return returns

def interact_without_showing(env, agent, num_episodes=20000, window_episodes=100):
    """ Monitor agent's performance.
    
    Params
    ======
    - env (obj): instance of the class MazeEnvironment
    - agent (obj): instance of class MazeAgent
    - num_episodes (int): number of episodes of agent-environment interaction

    Returns
    =======
    - returns: deque containing return of each episode
    """
    # Initialize a deque to save the average of rewards for each episode required
    returns = deque(maxlen=num_episodes)
    # for each episode
    for i_episode in range(1, num_episodes+1):
        # start it, getting the initial state of agent
        state = env.reset()
        # get first action of agent
        action = agent.start(state)
        # initialize the sampled return
        sample_return = 0.
        while True:
            # apply the actio choosen by agent
            (reward, next_state, done) = env.step(action)
            # update the sampled return
            sample_return += reward
            # update the state (s <- s') to next time step
            state = next_state
            # Check if the signal of termination is True
            if done:
                # Execute the last update of q-values and policy inside of agent
                agent.end(reward)
                # save sample_return in returns
                returns.append(sample_return)
                break
            # otherwise
            else:
                # get the next action and update the q-values and policy inside of agent
                action = agent.step(reward, next_state)
        # monitor progress
        print("\rEpisode {}/{} || return: {}".format(i_episode, num_episodes, sample_return), end="")
        sys.stdout.flush()
        if i_episode == num_episodes: print('\n')
    # close window
    env.close_window()
    return returns

# Test function
def main():
    # import MazeEnvironment and MazeAgent
    from MazeEnvironment import MazeEnvironment
    from MazeAgents import MazeQAgent
    import matplotlib.pyplot as plt
    # Define maze environment
    height = 6
    width = 6
    env_info = {'height':height, 'width':width, 'start_loc':(0,0),'goal_loc':(5,5), 'obstacles':[(2,2),(3,3)]}
    mazeEnv = MazeEnvironment(env_info)
    # define maze agent
    num_states = height*width
    num_actions = 9
    agent_info = {'num_states':num_states, 'num_actions':num_actions}
    qAgent = MazeQAgent(agent_info)
    # Execute episodes of interactions agent-environment
    num_episodes = 3000
    returns = interact(mazeEnv, qAgent, num_episodes)
    # plot result
    x = np.arange(1, num_episodes+1) # item number of episodes
    y = returns # average of returns from the window_episodes-th episode
    plt.plot(x,y)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Avg Returns')
    plt.show()

if __name__ == '__main__':
    main()