import gymnasium as gym
env = gym.make("ALE/Pong-v5")
observation, info = env.reset(seed=42)
# gym.make()


for _ in range(1000):
    # print('actions:')
    # print(env.action_space.sample())
    action = env.action_space.sample() # this is where you would insert your policy
    # print(env.step( action))
    observation, reward, terminated, truncated, info = env.step( action)
    print(f"reward: {reward} for action {action}")
    if terminated or truncated:
        observation, info = env.reset()


env.close()