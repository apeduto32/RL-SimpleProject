# Import packages
import sys
import time
import numpy as np

if sys.version_info == 2:
    import Tkinter as tk
else:
    import tkinter as tk

from BaseEnvironment import BaseEnvironment
from BaseAgent import BaseAgent

# Implement the class MazeEnvironment based on the mother class BaseEnvironment
class MazeEnvironment(BaseEnvironment):
    # Constructor
    def __init__(self, env_info):
        """
        Method used to setup a maze environment
        params:
            - env_info (dict): Dictionary containing enough information to setup a maze environment
        """

        # Initialize the unit (number of pixels), height and width (number of units) of maze environment
        self.unit = env_info.get('unit', 40)
        self.height_unit = env_info.get('height', 10)
        self.width_unit = env_info.get('width', 10)

        # check that grid_height and grid_width both are positive, for the case in which such fields are
        # setted by the input env_info not empty
        assert self.height_unit > 0,'grid_height is not positive'
        assert self.width_unit > 0,'grid_width is not positive'

        # Initialize the list of obstacles in maze if required
        self.obstacles = env_info.get('obstacles', [])

        # Initialize dictionary for actions by agent
        # Define a map (dictionary) having as keys integer values and as values bidimensional actions
        self.map_actions = {0:(0,0), # no movement
                             1: (0,1), # up
                             2: (1,1), # up-right
                             3: (1,0), # right
                             4: (1,-1), # down-right
                             5: (0,-1), # down
                             6: (-1,-1), # down-left
                             7: (0,-1), # left
                             8:(-1,1)} # up-left

        # Set number of actions attribute
        self.num_actions = len(self.map_actions)

        # Initialize the starting and goal position
        self.start_loc = env_info.get('start_loc')
        self.goal_loc = env_info.get('goal_loc')

        # Initialize the tuple attribute reward_state_term
        reward = None
        state = None
        done = False # Boolean used to indicate if the goal has been reached (True) or not (False)
        self.reward_state_done = (reward, state, done)

        # Create the main window for the environment
        self.window = tk.Tk()
        # set its title
        self.window.title('Maze')
        # set its geometry
        self.window.geometry('{}x{}'.format((self.width_unit+1)*self.unit, (self.height_unit+1)*self.unit))
        # Set the time of sleep between interactions with an agent
        self.time_sleep = env_info.get('time_sleep',0.01)
        # Build maze inside of the main window
        self.__build_maze()


    # Method used to build a maze
    def __build_maze(self):
        """
        Private Method used to create an interface of a maze environment
        """
        # Create a window of rectangular area by using canvas
        self.canvas = tk.Canvas(master=self.window, bg='white',
                                width=self.width_unit*self.unit, height=self.height_unit*self.unit)
        # draw vertical lines
        for c in range(0, self.width_unit*self.unit, self.unit):
            x0, y0, x1, y1 = c, 0, c, self.height_unit*self.unit
            self.canvas.create_line(x0, y0, x1, y1)
        # draw horizontal lines
        for c in range(0, self.height_unit*self.unit, self.unit):
            x0, y0, x1, y1 = 0, c, self.width_unit*self.unit, c
            self.canvas.create_line(x0, y0, x1, y1)

        # Create the goal point of agent
        goal_center = np.array([self.goal_loc[0], self.goal_loc[1]])*self.unit + self.unit/2.
        self.goal_point = self.canvas.create_rectangle(goal_center[0] - self.unit/2., goal_center[1] - self.unit/2.,\
                                                 goal_center[0]+self.unit/2.,goal_center[1]+self.unit/2., \
                                                 fill='green')

        # Create the starting point of agent
        start_center = np.array([self.start_loc[0], self.start_loc[1]])*self.unit + self.unit/2.
        self.start_point = self.canvas.create_rectangle(start_center[0]-self.unit/2., \
                                                            start_center[1]-self.unit/2.,\
                                                            start_center[0]+self.unit/2.,\
                                                            start_center[1]+self.unit/2.,\
                                                            fill='orange')
        # Create an agent in the starting point
        self.agent_point = self.canvas.create_oval(start_center[0]-self.unit/3., start_center[1]-self.unit/3.,\
                                                  start_center[0]+self.unit/3., start_center[1]+self.unit/3.,\
                                                  fill='red')


        # Create the obstacles if required
        if len(self.obstacles) > 0:
            self.obstacles_points = []
            for obstacle in self.obstacles:
                obstacle_center = np.array([obstacle[0], obstacle[1]])*self.unit + self.unit/2.
                self.obstacles_points.append(self.canvas.create_rectangle(obstacle_center[0]-self.unit/2., \
                                                                          obstacle_center[1]-self.unit/2.,\
                                                                          obstacle_center[0]+self.unit/2.,\
                                                                          obstacle_center[1]+self.unit/2., fill='black'))


        self.canvas.pack()

    def render(self):
        """
        Method used to show and to update the window for maze
        """
        # Wait
        time.sleep(self.time_sleep)
        # show and update window
        self.window.update()


    def __get_state_as_integer(self, state):
        """
        Method used to convert the agent's position (x,y) on an integer
        Params:
            - state --> tuple (x,y)
        Returns:
            - num_state --> integer value representing the input state
        """
        # Extract coordinates
        x = state[0]
        y = state[1]
        # check if such coordinate values corresponds to a position inside of the grid defined
        assert (x < self.width_unit and x >= 0 and y < self.height_unit and y >= 0),\
        'Position {} not inside of the Environment'.format(state)
        # Compute the converted state
        num_state = y*self.width_unit + x
        return num_state


    def reset(self):
        """
        Method used to set the start location of agent and return it as an integer.
        It's used also to set reward and termination term
        Params: None
        Returns: 
            - num_state --> integer corresponding to the starting state
        """
        # Initialize reward and done
        reward = None
        done = False
        # Move the agent to the starting point in the window
        start_center = np.array([self.start_loc[0], self.start_loc[1]])*self.unit + self.unit/2.0
        self.canvas.coords(self.agent_point, start_center[0]-self.unit/3., start_center[1]-self.unit/3., start_center[0]+self.unit/3., start_center[1]+self.unit/3.)
        # set the attribute agent_loc
        self.agent_loc = self.start_loc

        # get the corresponding number of the starting state
        num_state = self.__get_state_as_integer(self.agent_loc)
        self.reward_state_done = (reward, num_state, done)
        return num_state


    def step(self,action):
        """
        Method used to give a reward, a new state and termination signal when an action is taken by
        an agent.
        Params: action --> Integer representing an action by agent
        {stop: 0, up: 1, right-up: 2, right: 3, right-down: 4, down: 5, left-down: 6, left: 7, left-up: 8 }
        """
        # Get action of scape
        action2d = self.map_actions[action]
        # Extract components of action2d
        Ax = action2d[0]
        Ay = action2d[1]

        # Compute the next state in basic of the input action
        x = self.agent_loc[0]
        y = self.agent_loc[1]

        # Compute the next location of agent by remember the the limits of x and y
        next_loc = (max(0,min(x+Ax, self.width_unit-1)), \
                    max(0,min(y+Ay, self.height_unit-1)))

        # set reward to -1 and done to False
        reward = -1.
        done = False
        # check if agent_loc is equal to goal_loc
        if next_loc == self.goal_loc:
            # set reward to 1 and done to True
            reward = 1.
            done = True
        # otherwise check if the next location is an obstacle
        elif next_loc in self.obstacles:
            # set reward to -100
            reward = -100.
            # reset next_loc equal to the previous values
            next_loc = (x, y)
        # check if next loc is not equal to agent loc  
        if next_loc != self.agent_loc:    
            # Compute the components of effective movement for the case in which the choose action
            # will bring the agent outside the waze
            deltax, deltay = next_loc[0] - self.agent_loc[0], next_loc[1] - self.agent_loc[1]
            # update agent_loc
            self.agent_loc = next_loc
            # update position of agent in the window
            self.canvas.move(self.agent_point, deltax*self.unit, deltay*self.unit)


        # update the attribute reward_state_term
        state = self.__get_state_as_integer(self.agent_loc)
        self.reward_state_done = (reward, state, done)

        return self.reward_state_done

    def close_window(self):
        """
        Method used to close window
        """
        self.window.quit()
        self.window.destroy()
        self.window.mainloop()

    def cleanup(self):
        """
        Method used to clean a few attributes
        """
        pass

def main():
    # Define the environment
    env_info = {'height':6, 'width':6, 'start_loc':(0,0),'goal_loc':(5,5), 'obstacles':[(2,2),(3,3)]}
    maze = MazeEnvironment(env_info)
    # define the number of actions
    num_actions = 9
    # Starting
    maze.reset()
    maze.render()
    # for each step
    for _ in range(100):
        # choose random action
        action = np.random.choice(num_actions)
        # execute action
        (reward, state, done) = maze.step(action)
        # update window
        maze.render()
        print('action:{}, reward:{}, state:{}, done:{}'.format(action, reward, state, done))

    # close maze window
    time.sleep(3)
    maze.close_window()

if __name__ == '__main__':
    main()

