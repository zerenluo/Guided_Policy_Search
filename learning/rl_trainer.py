import pickle
import os
import numpy as np
import learning.utils as utils
from learning.tf_utils import ILNetwork
from learning.utils import PathDict
from typing import List
import copy

from learning.policy import MLPolicy
from learning.policy import NMPCCGMRESPolicy
from learning.policy import iLQRPolicy


from logging import getLogger
logger = getLogger(__name__)


import argparse

from PythonLinearNonlinearControl.helper import bool_flag, make_logger
from PythonLinearNonlinearControl.controllers.make_controllers import make_controller
from PythonLinearNonlinearControl.planners.make_planners import make_planner
from PythonLinearNonlinearControl.configs.make_configs import make_config
from PythonLinearNonlinearControl.models.make_models import make_model
from PythonLinearNonlinearControl.envs.make_envs import make_env
from PythonLinearNonlinearControl.runners.make_runners import make_runner
from PythonLinearNonlinearControl.plotters.plot_func import plot_results, \
    save_plot_data
from PythonLinearNonlinearControl.plotters.animator import Animator


class IL_trainer(object):
    def __init__(self, params):
        self.params = params
        self.env = make_env(params)
        self.IL = ILNetwork(params)
        self.collect_policy = MLPolicy(self.IL, params)

        if self.params.env == "CartPole":
            self.expert_policy = iLQRPolicy()

        elif self.params.env == "TwoWheeledTrack":
            self.expert_policy = NMPCCGMRESPolicy()

        self.x_training_data = np.zeros((1, self.env.config['state_size']), dtype=np.float32)
        self.u_training_data = np.zeros((1, self.env.config['input_size']), dtype=np.float32)


    def collect_training_trajectories(self, itr:int,
                                      load_initial_expertdata_path: str,
                                      traj_steps: int) -> List[PathDict]:
        paths: List[PathDict]
        print("\nCollecting data to be used for training...")

        loaded_path = PathDict(state=np.zeros((1, self.env.config['state_size']), dtype=np.float32),
                               action=np.zeros((1, self.env.config['input_size']), dtype=np.float32),
                               cost=0.)

        if itr == 0:
            load_initial_expertdata_state = os.path.join(load_initial_expertdata_path, self.params.env + "-history_x.pkl")
            with open(load_initial_expertdata_state, 'rb') as paths_file_state:
                loaded_path_state = pickle.load(paths_file_state)
                loaded_path['state'] = loaded_path_state

            load_initial_expertdata_action = os.path.join(load_initial_expertdata_path, self.params.env + "-history_u.pkl")
            with open(load_initial_expertdata_action, 'rb') as paths_file_action:
                loaded_path_action = pickle.load(paths_file_action)
                loaded_path['action'] = loaded_path_action

            load_initial_expertdata_cost = os.path.join(load_initial_expertdata_path, self.params.env + "-cost.pkl")
            with open(load_initial_expertdata_cost, 'rb') as paths_file_cost:
                loaded_path_cost = pickle.load(paths_file_cost)
                loaded_path['cost'] = loaded_path_cost

            paths = [loaded_path] # there is only one loaded expert path
        else:
            # paths = utils.sample_n_trajectories(self.env,
            #                                     self.collect_policy,
            #                                     batch_size // self.params['ep_len'],
            #                                     self.params['ep_len'])

            paths = utils.sample_n_trajectories(self.env,
                                                self.collect_policy,
                                                1,
                                                traj_steps)
        return paths

    def do_relabel_with_expert(self, paths: List[PathDict]) -> List[PathDict]:
        print("\nRelabelling collected observations with labels from an expert policy...")

        # TODO relabel collected obsevations (from our policy) with labels from an expert policy
        # IDEA: query the policy (using the get_action function) with paths[i]["observation"]
        # and replace paths[i]["action"] with these expert labels
        relabeled_paths: List[PathDict] = []

        for path in paths:
            relabeled_path = copy.deepcopy(path)
            for t, observation in enumerate(path['state']):
                path['action'][t] = self.expert_policy.get_action(observation)[np.newaxis, :] # right hand side returns shape(1, input_size)
                # print("\nGetting action from benchmark policy at time: ", t)
            relabeled_paths.append(relabeled_path)

        print("relabeled paths len:", len(paths))
        print("relabeled one path len:", paths[0]['state'].shape)

        return paths


    def run_training_loop(self, n_iter, initial_expertdata=None,
                          relabel_with_expert = False, start_relabel_with_expert=1,
                          traj_steps=0):
        """
        :param n_iter:  number of (dagger) iterations
        :param relabel_with_expert:  whether to perform dagger
        """

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(
                itr,
                initial_expertdata,
                traj_steps
            )
            paths = training_returns

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                paths = self.do_relabel_with_expert(paths)

            # add collected data to replay buffer, simply consider the buffer could be infinite large
            #     self.agent.add_to_replay_buffer(paths)
            for path in paths:
                if itr == 0:
                    self.x_training_data = path['state']
                    self.u_training_data = path['action']

                else:
                    self.x_training_data = np.concatenate((self.x_training_data, path['state']), axis=0)
                    self.u_training_data = np.concatenate((self.u_training_data, path['action']), axis=0)
                    print("after dagger the shape of self.x_training_data:", self.x_training_data.shape)
                    print("after dagger the shape of self.u_training_data:", self.u_training_data.shape)

            # train agent (using sampled data from replay buffer)
            self.train_agent()


    def train_agent(self):
        # start the training process
        batch_size = 128
        for training_step in range(self.params.num_train_steps_per_iter):
            # get the subset of the training datas
            indices = np.random.randint(low=0, high=len(self.x_training_data), size=batch_size)
            input_batch = self.x_training_data[indices]
            output_batch = self.u_training_data[indices]

            # TODO: use the train function in tf.utils: ILNetwork
            loss = self.IL.learn(input_batch, output_batch)

            if training_step % 1000 == 0:
                # print('training steps: ', training_step)
                print('training steps: {0: 04d}loss: {1:.5f}'.format(training_step, loss))

    def run(self, planner):
        """
        Returns:
            history_x (numpy.ndarray): history of the state,
            shape(episode length, state_size)
            history_u (numpy.ndarray): history of the state,
            shape(episode length, input_size)
        """

        done = False
        curr_x, info = self.env.reset()
        # print(curr_x.shape)
        history_x, history_u, history_g = [], [], []
        step_count = 0
        total_cost = 0.

        while not done:
            logger.debug("Step = {}".format(step_count))
            # plan, for const_planner: Returns replication of g_xs: g_xs (numpy.ndarrya): goal state, shape(pred_len, state_size)
            g_xs = planner.plan(curr_x, info["goal_state"])

            # obtain sol
            u = self.IL.forward(curr_x[np.newaxis, :]).reshape((self.env.config['input_size'],))

            # step
            next_x, cost, done, info = self.env.step(u)

            # save
            history_u.append(u)
            history_x.append(curr_x)
            history_g.append(g_xs[0])

            # update
            curr_x = next_x
            total_cost += cost
            step_count += 1

        logger.debug("Controller type = {}, Cost = {}"
                     .format('Learning Policy', total_cost))
        return np.array(history_x), np.array(history_u), np.array(history_g), total_cost


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--env", type=str, default="TwoWheeledTrack")
    parser.add_argument("--env", type=str, default="CartPole")

    parser.add_argument("--save_anim", type=bool_flag, default=1)
    # parser.add_argument("--controller_type", type=str, default="NMPCCGMRES")
    parser.add_argument("--controller_type", type=str, default="MPPI")

    parser.add_argument("--result_dir", type=str, default="./result")
    parser.add_argument("--use_learning", type=str, default=True)
    parser.add_argument("--num_train_steps_per_iter", type=str, default=100000)
    parser.add_argument("--relabel_with_expert", type=str, default=True)

    args = parser.parse_args()

    trainer = IL_trainer(args)
    initial_expertdata_path = os.path.join(args.result_dir + '/' + args.controller_type)
    print('Running rl_trainer for {0} with controller type {1}'.format(args.env, args.controller_type))

    # config = make_config(args)
    traj_steps = trainer.env.config["max_step"] # make sure that the sampled trajectory length is the same as the initial training trajectory length

    trainer.run_training_loop(n_iter=1, initial_expertdata=initial_expertdata_path, relabel_with_expert=args.relabel_with_expert,
                              start_relabel_with_expert=1, traj_steps=traj_steps)

    print('Finish training ...')

    # still testing, need to refactor
    make_logger(args.result_dir)
    config = make_config(args)
    planner = make_planner(args, config)

    history_x, history_u, history_g, cost = trainer.run(planner)
    plot_results(history_x, history_u, history_g=history_g, args=args)
    save_plot_data(history_x, history_u, history_g=history_g, cost=cost, args=args)

    if args.save_anim:
        animator = Animator(env=trainer.env, args=args)
        animator.draw(history_x, history_g)


if __name__ == "__main__":
    main()