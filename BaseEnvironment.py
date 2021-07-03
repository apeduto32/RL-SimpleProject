# Import packages to create an abstract class
from abc import ABC,abstractmethod

# Define the abstract class of a basic environment
class BaseEnvironment(ABC):
    """
    Abstarct class that defines the main methods of a generic environment
    """

    # Contructor
    @abstractmethod
    def __init__(self, env_info = {}):
        """
        Method used to setup an environment.
        """

    # Abstract method to start a cycle of interactions with an agent
    @abstractmethod
    def reset(self):
        """
        Method called to start a cycle of interactions with an agent
        """

    # Abstract method for a step of an environment
    @abstractmethod
    def step(self, action):
        """
        Method used by an environment to execute a step, given the action
        choosen by an agent
        """

    # Abstract method to cleanup the attributes of environment
    @abstractmethod
    def cleanup(self):
        """
        Method used to cleanup the attributes of an environment
        """
