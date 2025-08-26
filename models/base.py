import os
os.environ["JAX_ENABLE_X64"] = "true"
from abc import ABC, abstractmethod

class BaseModel(ABC):

    def __init__(self, params):
        self.params = params

    @abstractmethod
    def run(self, initial_state, n_steps, noise):
        pass