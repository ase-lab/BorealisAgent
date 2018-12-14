from . import BaseAgent
from .. import constants
import random
import numpy as np


class StaticAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""
    def __init__(self, *args, **kwargs):
        super(StaticAgent, self).__init__(*args, **kwargs)

    def act(self, obs, action_space):
        return constants.Action.Stop.value
