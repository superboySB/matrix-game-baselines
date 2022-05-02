from functools import partial

from .multiagentenv import MultiAgentEnv

from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
