import numpy as np
from learning.tf_utils import ILNetwork

# the env, config, planner all for CartPole and iLQR case
from PythonLinearNonlinearControl.envs.two_wheeled import TwoWheeledTrackEnv
from PythonLinearNonlinearControl.configs.two_wheeled import TwoWheeledExtendConfigModule
from PythonLinearNonlinearControl.planners.closest_point_planner import ClosestPointPlanner
from PythonLinearNonlinearControl.models.two_wheeled import TwoWheeledModel
from PythonLinearNonlinearControl.controllers.nmpc_cgmres import NMPCCGMRES

# class BasePolicy(object):
#     def get_action(self, obs: np.ndarray) -> np.ndarray:
#         raise NotImplementedError

class MLPolicy(object):
    def __init__(self, ILNet):
        self.ILNet = ILNet

    def get_action(self, curr_x: np.ndarray) -> np.ndarray:
        # this get_action is used for sampling training trajectory; namely collect_policy
        act = self.ILNet.forward(curr_x[np.newaxis, :]).reshape((2,))
        return act

class NMPCCGMRESPolicy(object):
    def __init__(self):
        self.env = TwoWheeledTrackEnv()
        self.config = TwoWheeledExtendConfigModule()
        self.planner = ClosestPointPlanner(self.config)
        self.model = TwoWheeledModel(self.config)
        self.controller = NMPCCGMRES(self.config, self.model)

    def get_action(self, curr_x: np.ndarray) -> np.ndarray:
        # this get_action is used for relabelling the state explored by the trained policy
        # plan, for const_planner: Returns replication of g_xs: g_xs (numpy.ndarrya): goal state, shape(pred_len, state_size). Here update the curr_x as current state
        curr_x, info = self.env.reset(curr_x)

        g_xs = self.planner.plan(curr_x, info["goal_state"])

        # obtain sol
        act = self.controller.obtain_sol(curr_x, g_xs)
        return act


