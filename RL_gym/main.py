import gym
import gym_env
import globe



env = gym.make('ISAC-ENV-v0')

# Plot the antenna UPA
env.plotSystem()


state = env.reset()
#
# for _ in range(10):
#     action = env.action_space.sample()  # 随机动作
#     state, reward, done, _ = env.step(action)
#     env.render()
#     if done:
#         break
# env.close()

for _ in range(10):  # 假设运行 10 步
    env.step(None)
env.plot_positions()
