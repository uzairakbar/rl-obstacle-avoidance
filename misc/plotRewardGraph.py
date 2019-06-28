# Load libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if len(sys.argv) < 4:
    print("Too few arguments. Please specify #simulations, #episodes/simulation and a list of base filenames.")
    quit()

num_simulations = int(sys.argv[1])
num_episodes = int(sys.argv[2])
episode_array = np.linspace(1, num_episodes + 1, num_episodes)

colors = cm.rainbow(np.linspace(0, 1, len(sys.argv[3:])))
fig = plt.figure()

for i, base_filename in enumerate(sys.argv[3:]):
	rewards = np.zeros((num_episodes, num_simulations))

	for sim_num in range(num_simulations):
		filename = "trainingRewardLog_" + str(sim_num) + base_filename + ".npy"
		rewards[:, sim_num] = np.load(filename)[:num_episodes]

	# Create means and standard deviations of rewards
	rewards_mean = np.mean(rewards, axis=1)
	rewards_std = np.std(rewards, axis=1)

	plt.plot(episode_array, rewards_mean, '--', color=colors[i],  label=base_filename)

	# Draw bands
	plt.fill_between(episode_array, rewards_mean - rewards_std, rewards_mean + rewards_std, color=colors[i], alpha = 0.5)

# Create plot
# plt.title("Learning Curve")
plt.xlabel("Number of Episodes"), plt.ylabel("Episode Reward")
plt.tight_layout()
plt.show()

fig.savefig("plotRewardGraph", format="pdf", dpi=1200)