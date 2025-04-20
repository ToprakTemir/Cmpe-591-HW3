from homework3 import *

env = Hw3Env(render_mode="gui")

agent = REINFORCE_Agent()

agent.load_model("/Users/toprak/cmpe591.github.io/src/new_hw3/src/reinforce_models/model_20250417-234458")

while True:
    obs = env.reset()
    done = False
    step_count = 0
    cum_reward = 0

    while not done:
        action, _ = agent.predict(torch.from_numpy(obs).float())
        obs, reward, terminated, truncated = env.step(action)
        cum_reward += reward
        done = terminated or truncated
        step_count += 1
        env.viewer.render()

        if done:
            print(f"Episode finished after {step_count} timesteps, reward: {cum_reward}")
            break
