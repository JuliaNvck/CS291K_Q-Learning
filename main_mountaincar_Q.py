import argparse

import numpy as np

from mountaincar import MountainCar, FeatureMap
import a3_student as student


def main(genmodel: bool, sparse: bool, seed: int, render=False, iters=5000):
    """Top-level function to run Q-learning and return success/failure data.

    Args:
        genmodel (bool): If true, use generative model sampling. Otherwise, use
            epsilon-greedy.
        sparse (bool): Sparse reward flag. See mountaincar.py and handout for info.
        seed (int): PRNG seed for reproducibility.
        render (bool): If true, shows graphics window for greedy rollouts.
        iters (int): Number of iterations (1 iteration = 1 batch gradient update).

    Returns:
        greedy_goals (List[Tuple[bool, int]]): For each greedy rollout
            evaluation, pair of 1. did the policy find the goal, and 2. if so,
            after how many actions?
    """

    # Params used everywhere.
    N_ENVS = 1000
    GAMMA = 0.98
    TIMELIMIT = 200

    mc = MountainCar(n_envs=N_ENVS, sparse_reward=sparse, seed=0)
    featuremap = FeatureMap()

    # A second instance for visualizing greedy rollouts, so we don't mess with
    # the state of the main one.
    mc_test = MountainCar(n_envs=1, sparse_reward=sparse, seed=0)

    # Initialize Q-learning state.
    example_feature = featuremap(mc.reset())[0]
    feature_dim = example_feature.size
    weights = np.zeros((feature_dim, mc.N_ACTIONS))
    target_weights = weights.copy()
    target_update_every = 1 if genmodel else TIMELIMIT

    # Store the results of each greedy evaluation.
    greedy_goals = []

    # Pick the method to use for generating samples of
    # (state, action, reward, is_terminal, next-state).
    if genmodel:
        sampler = student.genmodel_generate_samples(mc, seed)
        rate = 0.3
    else:
        sampler = student.epsilon_greedy_generate_samples(
            mc,
            timelimit=TIMELIMIT, epsilon=0.1,
            featuremap=featuremap, weights=weights,
            seed=seed,
        )
        rate = 0.03

    # Main loop.
    for iter in range(iters):
        s, a, r, is_terminal, s_next = next(sampler)

        student.compute_batch_update_Q(
            featuremap,
            weights, target_weights,
            s, a, r, is_terminal, s_next,
            gamma=GAMMA, weight_decay=1e-2, learning_rate=rate,
        )

        if iter % target_update_every == 0:
            target_weights = weights.copy()

        if iter % 200 == 0:
            print(f"{iter = }")
            finished, time = student.rollout_greedy(
                mc_test, TIMELIMIT, featuremap, weights, render=render)
            if finished:
                print(f"done in {time} steps.")
            greedy_goals.append((finished, time))

    mc.close()
    mc_test.close()
    return greedy_goals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Q-learning variants in MountainCar.")
    parser.add_argument("--genmodel", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    main(args.genmodel, args.sparse, args.seed, args.render)
