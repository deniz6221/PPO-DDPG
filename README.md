## PPO and DDPG Implementation
This project implements DDPG and PPO to solve the LunarLanderContinuous-v3 environment.

### DDPG
The DDPG model is trained in 2 stages for it to be successful. The agent only learns how to fly at the first stage with the default rewards. 
At the second stage, the reward function is slightly modified to punish staying still in the air. After this, the model improves significantly. \
The reward function with only one stage can be seen in DDPG/reward_plot.png, the final plot with two stages can be seen in DDPG/reward_two_stage.png \
\
<img src="https://github.com/deniz6221/PPO-DDPG/blob/main/DDPG/gifs/1.gif" alt="ddpg1" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">
<img src="https://github.com/deniz6221/PPO-DDPG/blob/main/DDPG/gifs/2.gif" alt="ddpg2" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px"> \
<img src="https://github.com/deniz6221/PPO-DDPG/blob/main/DDPG/gifs/3.gif" alt="ddpg3" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">
<img src="https://github.com/deniz6221/PPO-DDPG/blob/main/DDPG/gifs/4.gif" alt="ddpg4" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">

### PPO
A simple PPO algorithm without GAE is used with a slightly modified reward function that punishes staying still in the air. The model learned the
enviroment in 5000 episodes due to the powerfull nature of PPO. \
The reward plot can be seen in PPO/reward_plot.png \
\
<img src="https://github.com/deniz6221/PPO-DDPG/blob/main/PPO/gifs/1.gif" alt="ppo1" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">
<img src="https://github.com/deniz6221/PPO-DDPG/blob/main/PPO/gifs/2.gif" alt="ppo2" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px"> \
<img src="https://github.com/deniz6221/PPO-DDPG/blob/main/PPO/gifs/3.gif" alt="ppo3" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">
<img src="https://github.com/deniz6221/PPO-DDPG/blob/main/PPO/gifs/4.gif" alt="ppo4" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage" width="300px">

### How To Run

Install the required libraries first. (torch and gymnasium) \
To train a PPO model: \
1- cd to PPO `cd PPO` \
2- Run `python3 train.py` \
\
To test the PPO model: \
1- cd to PPO `cd PPO` \
2- Run `python3 test.py` \
\
To train a DDPG model: \
1- cd to DDPG `cd DDPG` \
2- Run `python3 train.py` \
\
To test the DDPG model: \
1- cd to DDPG `cd DDPG` \
2- Run `python3 test.py` 
