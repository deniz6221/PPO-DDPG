import numpy as np
import matplotlib.pyplot as plt
import json


rews = json.load(open("checkpoints/rews_10000.json"))
rews = np.array(rews)

window_size = 100

smoothed_rews = np.convolve(rews, np.ones(window_size)/window_size, mode='valid')

lossX = [i for i in range(len(smoothed_rews))]

plt.plot(lossX, smoothed_rews)
plt.title("Reward over episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.legend()
plt.show()