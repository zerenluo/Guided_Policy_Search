import numpy as np
from PythonLinearNonlinearControl.envs.make_envs import make_env

# the env, config, planner all for TwoWheeledTrack and NMPCCGMRES case
from PythonLinearNonlinearControl.envs.two_wheeled import TwoWheeledTrackEnv
from PythonLinearNonlinearControl.configs.two_wheeled import TwoWheeledExtendConfigModule
from PythonLinearNonlinearControl.planners.closest_point_planner import ClosestPointPlanner
from PythonLinearNonlinearControl.models.two_wheeled import TwoWheeledModel
from PythonLinearNonlinearControl.controllers.nmpc_cgmres import NMPCCGMRES

# the env, config, planner all for CartPole and iLQR case
from PythonLinearNonlinearControl.envs.cartpole import CartPoleEnv
from PythonLinearNonlinearControl.configs.cartpole import CartPoleConfigModule
from PythonLinearNonlinearControl.planners.const_planner import ConstantPlanner
from PythonLinearNonlinearControl.models.cartpole import CartPoleModel
from PythonLinearNonlinearControl.controllers.ilqr import iLQR


class MLPolicy(object):
    def __init__(self, ILNet, params):
        self.ILNet = ILNet
        self.env = make_env(params)

    def get_action(self, curr_x: np.ndarray) -> np.ndarray:
        # this get_action is used for sampling training trajectory; namely collect_policy
        act = self.ILNet.forward(curr_x[np.newaxis, :]).reshape((self.env.config['input_size'],))
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


class iLQRPolicy(object):
    def __init__(self):
        self.env = CartPoleEnv()
        self.config = CartPoleConfigModule()
        self.planner = ConstantPlanner(self.config)
        self.model = CartPoleModel(self.config)
        self.controller = iLQR(self.config, self.model)

    def get_action(self, curr_x: np.ndarray) -> np.ndarray:
        # this get_action is used for relabelling the state explored by the trained policy
        # plan, for const_planner: Returns replication of g_xs: g_xs (numpy.ndarrya): goal state, shape(pred_len, state_size). Here update the curr_x as current state
        curr_x, info = self.env.reset(curr_x)

        g_xs = self.planner.plan(curr_x, info["goal_state"])

        # obtain sol
        act = self.controller.obtain_sol(curr_x, g_xs)
        return act


