import os
import pickle
import numpy as np
import gym

# paths_file_u = os.path.join("./result", 'iLQR/' + "CartPole" + "-history_u.pkl")
#
# with open(paths_file_u, 'rb') as paths_file_action:
#     loaded_path_action = pickle.load(paths_file_action)
#
# print(loaded_path_action)
# # print(loaded_path_action.type)
#
# print(np.amax(loaded_path_action))
# print(np.amin(loaded_path_action))
#
#
# print(loaded_path_action.shape)


# import gym
# env = gym.make('InvertedPendulum-v2')
# print(env.action_space)
# #> Discrete(2)
# print(env.observation_space)
# print(isinstance(env.action_space, gym.spaces.Discrete))
# print(env.action_space.shape[0])
#
# env2 = gym.make('CartPole-v0')
# print(env2.action_space)
# #> Discrete(2)
# print(env2.observation_space)
# print(isinstance(env2.action_space, gym.spaces.Discrete))
# print(env2.action_space.shape[0])

# a = np.array([[[1, 2, 3],
#
# [4, 5, 6]],
#
# [[2, 2, 3],
#
# [4, 5, 6]],
#
# [[3, 2, 3],
#
# [4, 5, 6]]])
# # a = np.array([[1,2,3],[4,5,6]])
#
#
#
# print(a.shape)
#
# b = a.reshape([-1,3])
# print(b.shape)
# print(b)

x_init = 2 * np.random.rand(1) - 1
print(x_init)

