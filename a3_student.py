import numpy as np

from mountaincar import MountainCar

"""Assignment-wide conventions:

This notation is used throughout docstrings:
    N = number of batched environments.
    S = dimensionality of state space.
    A = number of discrete actions.
    d = feature dimension.

The following arguments and/or return values always mean:

    featuremap (callable): Function that maps array (N, S) of states to array
        (N, d) of feature vectors. Note: Do not assume it is a
        mountaincar.FeatureMap; autograder uses other featuremaps too.

    weights (array(d, A)): Weights for linear approximation of the Q-function.
        The appoximation of Q(s, a) is the inner product of featuremap(s) and
        weights[:, a].
"""


def compute_td_target(rewards, gamma, is_terminals, next_Qs):
    """Computes TD targets for Q-learning updates.

    Args:
        rewards (array(N)): Rewards received for the batch of transitions.
        gamma (float): Discount factor.
        is_terminals (array of bool): Boolean array indicating whether or not the state is terminal.
        next_Qs (array(N, A)): Q-values for each action at the next states.
        TD-target = r + gamma(1 - tau) max_{a'} Q_{theta_{target}}(s', a')

    Returns:
        td_targets (array(N)): TD targets following the expression from the handout.
    """
    td_targets = rewards + gamma * (1 - is_terminals) * np.max(next_Qs, axis=1)
    return td_targets


def compute_batch_update_Q(
    featuremap,
    weights,
    target_weights,
    states,
    actions,
    rewards,
    is_terminals,
    next_states,
    gamma,
    weight_decay,
    learning_rate,
):
    """Performs a batch Q-learning update with linear function approximation.

    This function should UPDATE the `weights` array in-place. See the handout
    for more details on the expected behavior.

    Note: Obtain dimensions from the input shapes; do not assume they are the
    MountainCar dimensions. Autograder uses some other dimensions too.

    Args:
        featuremap (callable): As in conventions.
        weights (array(d, A)): As in conventions. Weights to be updated
            IN-PLACE by the function.
        target_weights (array(d, A)): As in conventions. Target weights to use
            in constructing the TD-target. Should not be updated.
        states (array(N, S)): Batch of N states.
        actions (array(N) of int): Actions taken in each state.
        rewards (array(N)): Rewards received.
        is_terminals (array(N) of bool): Batch of is-terminal flags.
        next_states (array(N, S)): Batch of next states.
        gamma (float): Discount factor.
        weight_decay (float): L2 regularization coefficient.
        learning_rate (float): Step size for gradient update.

        $$J(theta) = 1/|D| * sum 1/2 (TD_Target} - Q_{theta}(s,a))^2

    Returns:
        None. The function modifies `weights` in-place.
    """
    phi_s = featuremap(states) # (N, d)
    current_Qs = phi_s @ weights
    current_Qs = current_Qs[np.arange(len(actions)), actions]
    next_Qs = featuremap(next_states) @ target_weights
    td_targets = compute_td_target(rewards, gamma, is_terminals, next_Qs)
    features = phi_s
    errors = td_targets - current_Qs
    one_hot_actions = np.zeros((len(actions), weights.shape[1]))  # (N, A)
    one_hot_actions[np.arange(len(actions)), actions] = 1 # make one-hot encoding of actions
    routed_errors = one_hot_actions * errors[:, np.newaxis] # (N, A) array where only the column corresponding to the action taken has the error, others are zero
    gradient = -(features.T @ routed_errors) / len(actions) # (d, A) gradient of the loss with respect to weights
    # update weights with gradient descent step and weight decay    
    weights -= learning_rate * (gradient)
    # apply weight decay
    weights -= learning_rate * weight_decay * weights
    return None



def genmodel_generate_samples(mc, seed):
    """Generator that yields uniform random samples from the MDP transition model.

    See handout for details on expected behavior.
    uniform sample over S x A,
    s ~ Uniform(S), a ~ Uniform(A)
    Use the entire_state_space option in MountainCar.reset to enable this. Note that each
    dataset is totally independent, and this sampler does not need any knowledge of the current θ. 

    Args:
        mc (MountainCar): MountainCar environment object.
        seed (int): PRNG seed for repeatability. NOTE: Use MountainCar.reseed()
            to ensure that MoutainCar.reset() is also reproducible.

    Yields:
        Infinite sequence of tuples of (s, a, r, is_terminal, s_next), where:
            s (array(N, S)): Batch of random states.
            a (array(N) of int): Batch of random actions.
            r (array(N)): Batch of rewards.
            is_terminal (array(N) of bool): Batch of is-terminal flags.
            s_next (array(N, S)): Batch of next states.
    """
    mc.reseed(seed)
    N = mc.n_envs # batch size - number of parallel environments
    while True:
        states = mc.reset(entire_state_space=True) # (N, S) array of random states
        actions = mc.rng.integers(0, mc.N_ACTIONS, size=N) # (N,) array of random actions
        next_states, rewards, is_terminals = mc.step(actions) # take a step in the environment with the random actions
        yield (
            states,
            actions,
            rewards,
            is_terminals,
            next_states,
        )


def epsilon_greedy_generate_samples(mc, timelimit, epsilon, featuremap, weights, seed):
    """Generator that yields samples from epsilon-greedy policy rollouts.

    See handout for details on expected behavior. Note: Use MountainCar's
    masked reset() at each step for environments where is_terminal is True.

    Note: The value of `weights` will be modified by the calls to
    compute_batch_update_Q that occur between each `yield` statement. Do not
    copy `weights`; always read from the provided array.

    For the parameter ϵ ∈ (0, 1), it chooses actions according to
        (
        a ~ Uniform(A) : with probability ϵ
        argmax (a∈A) Qθ(s, a) : otherwise

    It uses its current knowledge (the weights) to act optimally most of the time, 
    but occasionally takes a random action to discover new strategies.
    Because the agent is playing sequentially, this generator needs to remember the state between yield calls.

    Args:
        mc (MountainCar): MountainCar environment object.
        timelimit (int): Maximum number of steps per episode before full reset.
        epsilon (float): Probability of taking a random action.
        featuremap (callable): As in conventions.
        weights (array(d, A)): Current Q-function weights, as in conventions.
        seed (int): PRNG seed for reproducibility.

    Yields:
        Infinite sequence of tuples of (s, a, r, is_terminal, s_next), where:
            s (array(N, S)): Batch of random states.
            a (array(N) of int): Batch of random actions.
            r (array(N)): Batch of rewards.
            is_terminal (array(N) of bool): Batch of is-terminal flags.
            s_next (array(N, S)): Batch of next states.
    """
    mc.reseed(seed)
    N = mc.n_envs
    states = mc.reset() # initial state for each environment
    steps = 0 # to keep track of how many actions the agent has taken since the last full reset
    while True:
        current_Qs = featuremap(states) @ weights # (N, A) array of Q-values for each action in the current states
        greedy_actions = np.argmax(current_Qs, axis=1) # (N,) array of greedy actions for each environment
        random_actions = mc.rng.integers(0, mc.N_ACTIONS, size=N) # (N,) array of random actions for each environment
        explore = mc.rng.random(size=N) < epsilon # (N,) boolean array where True means take random action, False means take greedy action
        actions = np.where(explore, random_actions, greedy_actions) # (N,) array: if explore[i] is True, take random_actions[i], else take greedy_actions[i]
        next_states, rewards, is_terminals = mc.step(actions) # take a step in the environment with the chosen actions
        steps += 1
        yield (
            states,
            actions,
            rewards,
            is_terminals,
            next_states,
        )
        # If any environment is terminal or if we've reached the time limit, reset those environments
        if steps >= timelimit:
            states = mc.reset() # full reset of all environments
            steps = 0 # reset step count after a full reset
        elif np.any(is_terminals):
             # Masked reset only for environments that finished
            # (Unfinished environments automatically stay at next_states)
            states = mc.reset(mask=is_terminals)
        else:
            states = next_states # update state for the next iteration
        


def rollout_greedy(mc, timelimit, featuremap, weights, render=False):
    """Executes a single greedy rollout and returns success status.

    This is used for evaluation: it runs the greedy policy (no exploration)
    and checks if the agent reaches the goal within the time limit.

    Args:
        mc (MountainCar): MountainCar environment object.
        timelimit (int): Maximum number of steps before giving up.
        featuremap (callable): As in conventions.
        weights (array(d, A)): As in conventions.
        render (bool): If True, call mc.render() at each step for visualization.

    Returns:
        (success, steps): Tuple where:
            success (bool): True if the agent reached the goal (terminal state).
            steps (int): Number of steps to reach goal if successful, -1 if failed.
    """
    s = mc.reset()
    steps = 0
    while steps < timelimit:
        current_Qs = featuremap(s) @ weights # (N, A) array of Q-values for each action in the current states
        greedy_actions = np.argmax(current_Qs, axis=1) # (N,) array of greedy actions for each environment
        next_states, rewards, is_terminals = mc.step(greedy_actions) # take a step in the environment with the greedy actions
        steps += 1
        if render:
            mc.render()
        if np.any(is_terminals): # if any environment has reached the goal
            return True, steps
        s = next_states # update state for the next iteration
    return False, -1
