import random
import gym
import matplotlib.pyplot as plt
import numpy as np


UP_ACTION = 2
DOWN_ACTION = 3


env = gym.make("Pong-v0")
observation = env.reset()


# for i in range(22):
#     action = random.randint(2, 3) # this is where you would insert your policy
  
#     if i > 20:
#         plt.imshow(observation)
#         plt.show()


#     observation, reward, terminated, info = env.step(action)
#     print(f"reward: {reward} for action {action}")



for _ in range(1000):
    action = random.randint(UP_ACTION, DOWN_ACTION) # this is where you would insert your policy


    observation, reward, terminated, info = env.step(action)
    print(f"reward: {reward} for action {action}")
    # if terminated or truncated:
    #     observation, info = env.reset()


env.close()