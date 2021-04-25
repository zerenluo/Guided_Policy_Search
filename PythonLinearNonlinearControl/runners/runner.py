from logging import getLogger

import numpy as np

logger = getLogger(__name__)


class ExpRunner():
    """ experiment runner
    """

    def __init__(self):
        """
        """
        pass

    def run(self, env, controller, planner):
        """
        Returns:
            history_x (numpy.ndarray): history of the state,
            shape(episode length, state_size)
            history_u (numpy.ndarray): history of the state,
            shape(episode length, input_size)
        """
        done = False
        curr_x, info = env.reset()
        history_x, history_u, history_g = [], [], []             # this is the one sampled trajectory
        step_count = 0
        total_cost = 0.

        while not done:
            logger.debug("Step = {}".format(step_count))
            # plan, for const_planner: Returns replication of g_xs: g_xs (numpy.ndarrya): goal state, shape(pred_len, state_size)
            g_xs = planner.plan(curr_x, info["goal_state"])

            # obtain sol
            u = controller.obtain_sol(curr_x, g_xs)

            # step
            next_x, cost, done, info = env.step(u)

            # save
            history_u.append(u)
            history_x.append(curr_x)
            history_g.append(g_xs[0])
            # update
            curr_x = next_x
            total_cost += cost
            step_count += 1

        logger.debug("Controller type = {}, Cost = {}"
                     .format(controller, total_cost))
        return np.array(history_x), np.array(history_u), np.array(history_g), total_cost
