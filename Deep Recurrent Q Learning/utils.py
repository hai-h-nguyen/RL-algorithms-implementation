import numpy as np
import torch
import random

def set_global_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def rolling_average(data, *, window_size):
    """Smoothen the 1-d data array using a rollin average.

    Args:
        data: 1-d numpy.array
        window_size: size of the smoothing window

    Returns:
        smooth_data: a 1-d numpy.array with the same size as data
    """
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]


def render(test_env, policy=None):
    state = test_env.reset()
    test_env.render()

    total_sim = 100
    cnt_sim = 0
    ep_reward = 0

    cnt_reward = 0

    while cnt_sim < total_sim:
        action = policy(state)
        state, reward, done, _ = test_env.step(action)
        ep_reward += reward
        test_env.render()

        if done:
            cnt_sim += 1
            print(ep_reward)

            if ep_reward > 0:
                cnt_reward += 1

            ep_reward = 0
            state = test_env.reset()

    print("Test success rate: ", cnt_reward/total_sim)
    test_env.close()


def select_action_epsilon_greedy(dqn_model, state, eps, env):
    """Perform epsilon greedy action selection based on the Q-values.

    :param dqn_model: Q network
    :param eps: The probability to select a random action. Float between 0 and 1.

    Shapes:
        output: Scalar.
    """

    if np.random.rand() < eps:
        num_actions = env.action_space.n
        action = np.random.randint(num_actions)
    else:
        with torch.no_grad():
            state = torch.from_numpy(state)
            state = state.float()
            q_value = dqn_model(state)
            action = q_value.max(0)[1].item()

    return action