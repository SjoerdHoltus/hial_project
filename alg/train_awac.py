import numpy as np
from policy_learn import AWAC
from trajectory_utils import generate_trajectories_from_files
from init_env import init_env
from policy_learn import calculate_reward

def prepare_demonstrations(trajectories):
    states, actions, next_states, rewards, dones = [], [], [], [], []
    for traj in trajectories:
        for i in range(len(traj)-1):
            state = traj[i][0]
            action = traj[i][1]
            next_state = traj[i+1][0]
            reward = calculate_reward(state, next_state)
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(False)
            
        # Add final transition
        states.append(traj[-1][0])
        actions.append(traj[-1][1])
        next_states.append(traj[-1][0])
        rewards.append(0)
        dones.append(True)
    
    return np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones)

def main():
    
    # Initialize environment
    env = init_env(render=False)
    
    # Get demonstrations
    trajectories, _ = generate_trajectories_from_files()
    demo_states, demo_actions, demo_next_states, demo_rewards, demo_dones = prepare_demonstrations(trajectories)
    
    # Initialize AWAC
    state_dim = demo_states.shape[1]
    action_dim = demo_actions.shape[1]
    awac = AWAC(state_dim, action_dim)

    # Training loop
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action = awac.select_action(state)
            next_state, _, done, _ = env.step(action)
            
            # Calculate reward using learned weights
            reward = calculate_reward(state, next_state)
                
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        print(f"Episode {episode}: Reward = {episode_reward}")

if __name__ == "__main__":
    main()