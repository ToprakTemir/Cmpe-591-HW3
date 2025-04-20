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
![5000_ep_reinforce](https://github.com/user-attachments/assets/179c2b9e-bbed-4f1a-beb8-56cc2594156d)

### SAC reward plot:
![sac_rewards](https://github.com/user-attachments/assets/ae2c0973-6faa-44dd-bb16-078bffccaf4c)



## File and logs for the previous version of the project
The old version of the homework3 is implemented in the old_homework3 file, with collector, agent and main classes all in the same file.  
The old version is implementing a PPO for the same environment, over our current _homework3.py.

### reward plot:
The reward is 1/ee_to_obj + 1/obj_to_target  

![latest_plot](https://github.com/user-attachments/assets/24f669e7-68db-4a38-b202-e5ee2b1ecc74)
