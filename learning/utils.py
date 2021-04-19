import numpy as np
from typing_extensions import TypedDict
from typing import List


class PathDict(TypedDict):
    state: np.ndarray
    action: np.ndarray
    cost: np.float32

def sample_trajectory(
        env,
        policy,
        max_path_length: int )-> PathDict:

    # initialize env for the beginning of a new rollout
    x_init, info = env.reset()
    x: np.ndarray = x_init

    # init vars
    xs: List[np.ndarray] = []
    us: List[np.ndarray] = []
    next_xs: List[np.ndarray] = []
    total_cost = 0.

    steps = 0

    while True:
        # use the most recent ob to decide what to do
        xs.append(x)

        # TODO: policy here should be training policy, here plan a trajectory with training policy, input current state, return a series of actions along this trajectory
        u = policy.get_action(x)
        u = u[0]       # take the first action, MPC fashion
        us.append(u)

        # take that action and record results
        next_x, cost, done, info = env.step(u)

        # record result of taking that action
        steps += 1
        next_xs.append(next_x)
        total_cost += cost

        # end the rollout if the rollout ended. rollout can end due to done, or due to max_path_length
        rollout_done = bool(done) or steps >= max_path_length

        if rollout_done:
            break

    return Path(xs, us, total_cost)

def sample_n_trajectories(env, policy, ntraj: int, max_path_length: int) -> List[PathDict]:
    # Collect ntraj rollouts.
    paths: List[PathDict] = []

    for _ in range(ntraj):
        paths.append(sample_trajectory(env, policy, max_path_length))

    return paths

def Path(xs: List[np.ndarray], us: List[np.ndarray], cost: np.float32) -> PathDict:
    return {
        "state": np.array(xs, dtype=np.float32),
        "action": np.array(us, dtype=np.float32),
        "cost": cost
    }