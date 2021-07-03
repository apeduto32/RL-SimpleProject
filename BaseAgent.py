# Import package to define an abstract class
from abc import ABC, abstractmethod

# Define the abstract class of a basic agent
class BaseAgent(ABC):
    """
    Abstact class that defines the main methods of a generic agent.
    """

    # Constructor
    @abstractmethod
    def __init__(self, agent_info={}):
        """
        Method to setup an agent
        """

    # Method for the first step of an agent
    @abstractmethod
    def start(self, state):
        """
        Method used by an agent to choice the first action given the
        starting state as an input.
        """

    # Method for a generic step of an agent
    @abstractmethod
    def step(self, reward, next_state):
        """
        Method used by an agent in each step of interaction with an
        environment except for the first and the last ones
        """

    # Method for the final step of an agent
    @abstractmethod
    def end(self, reward):
        """
        Method used by an agent when the goal state has been reached
        """

    # Method used to cleanup memory of an agent
    @abstractmethod
    def cleanup(self):
        """
        Method used to cleanup the attributes of an agent
        """
