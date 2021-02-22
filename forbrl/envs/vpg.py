
from .base import BaseEnv
from .VolksEnv.environment.assembler import assemble
from ..utils import ENVIRONMENTS


@ENVIRONMENTS.register_module
class VPGEnv(BaseEnv):

    def __init__(self, env):
        self.game = assemble(env)
        self.game.connect()

    def step(self, action):
        reward = self.game.make_action(action)
        state = self.game.get_state()
        done = self.game.is_episode_finished()

        return state, reward, done

    def reset(self):
        self.game.new_episode()
        return self.game.get_state()

    def render(self, mode='sim'):
        pass

    def close(self):
        self.game.close()

    def __enter__(self):
        """Support with-statement for the environment. """
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment. """
        self.close()
        # propagate exception
        return False
