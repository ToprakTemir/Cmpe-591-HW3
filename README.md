# Cmpe-591-HW3

## How to run:
There are 2 main functions named "reinforce_main" and "sac_main" in the homework3.py file.  
Both main functions initialize the corresponding agent and train their respective models. The necessary logging is done during training.  


##Reward plots:
The set of environment hyperparameters I used for the environment:
- _goal_tresh = 0.040
- _max_timesteps = 200
- c_direction = 0.3021260995439765
- c_ee_to_obj = 0.27168714922932846
- c_obj_to_target = 0.127750450144418
- completion_reward = 4.547255827329526

### REINFORCE reward plot:
![reinforce_rewards](https://github.com/user-attachments/assets/e11e4852-6355-40e3-8a55-90707d798b19)


### SAC reward plot:
![sac_rewards](https://github.com/user-attachments/assets/2756b869-bd82-4350-aa0c-570a5463998c)




## File and logs for the previous version of the project
The old version of the homework3 is implemented in the old_homework3 file, with collector, agent and main classes all in the same file.  
The old version is implementing a PPO for the same environment, over our current _homework3.py.

### reward plot:
The reward is 1/ee_to_obj + 1/obj_to_target for 100 steps 

![latest_plot](https://github.com/user-attachments/assets/24f669e7-68db-4a38-b202-e5ee2b1ecc74)
