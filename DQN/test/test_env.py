import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import gym

env = gym.make("TwoBumps-v0")
obs = env.reset()
done = False

test_cnt = 0
total_tests = 200

while test_cnt <= total_tests:
    action = env.action_sample()
    obs, reward, done, _ = env.step(action)
    env.render()

    if done:
        env.reset()
        test_cnt += 1

env.close()