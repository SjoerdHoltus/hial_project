import os 
import numpy as np
import pybullet as p
from moviepy.editor import ImageSequenceClip
from init_env import init_env

def generate_trajectory(env, max_episode_length, save_dir, save_name, seed=None):
    """
    Generate a trajectory from the environment.
    """
    traj = []
    obs = env.reset()
    done = False
    t = 0
    env.action_space.seed(seed)
    while not done and t < max_episode_length:
        act = env.action_space.sample()
        traj.append((obs,act))
        obs, reward, done, info = env.step(act) # obs, reward, done, info
        t += 1
    traj.append((obs, None))
    env.close()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/{save_name}', exist_ok=True)
        with open(f'{save_dir}/{save_name}/action_traj.csv', 'w') as f:
            for _, act in traj:
                if act is None:
                    break
                f.write(f'{act[0]} {act[1]} {act[2]} {act[3]}\n')
        with open(f'{save_dir}/{save_name}/state_traj.csv', 'w') as f:
            for obs, _ in traj:
                line = ''.join([f"{list(obs['observation'])[i]} " for i in range(len(list(obs['observation']))-3)]) + f"{list(obs['achieved_goal'])[0]} {list(obs['achieved_goal'])[1]} {list(obs['achieved_goal'])[2]}\n"
                f.write(line)
    return traj

def record_trajectory(env, trajectory, starting_state=None):
    """
    Record a trajectory from the environment.
    """
    frames = []
    if not starting_state:
        env.reset() # If no starting state, reset environment to initial state
    else:
        # If starting state, reset environment to starting state. 
        # This is used to record expert trajectories (when we have a recorded starting state)
        env.reset(whether_random=False, object_pos=starting_state[7:10])
    
    for obs, act in trajectory:
        if act is None:
            break
        frame = np.uint8(env.render(mode='rgb_array'))
        frame = frame.reshape(480, 720, 4)
        frame = frame[:, :, :3]
        frames.append(frame)
        env.step(act)
    
    env.close()
    frames = np.array(frames)
    return frames

def generate_clip(frames, filename, fps=30):
    """
    Generates a clip from a recording of a trajectory.
    """
    if len(frames) == 0:
        print("No frames captured!")
        return
    print(f"Generating clip with {len(frames)} frames")
    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_videofile(filename, codec="libx264")
    return

def generate_trajectories_from_files():
    """
    Function to generate trajectories from the expert demonstrations.
    """
    import os
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR) 
    os.chdir(PARENT_DIR)
    os.chdir('demo_data/PickAndPlace')
    action_file_path = 'action_traj.csv'
    state_file_path = 'state_traj.csv'
    trajectories = []
    traj = []
    
    with open(action_file_path, 'r') as f:
        with open(state_file_path, 'r') as g:
            for action, state in zip(f, g):
                if 'inf' in action:
                    if traj:  # If we have a trajectory, append it
                        trajectories.append(traj)
                    traj = []  # Start new trajectory
                    continue  # Skip the inf line
                
                # Process normal lines
                state = state.strip()
                action = action.strip()
                state = [float(s) for s in state.split(' ')]
                action = [float(a) for a in action.split(' ')]
                traj.append((state, action))
            
            # Append the last trajectory
            if traj:
                trajectories.append(traj)
    
    starting_states = [traj[0][0] for traj in trajectories]
    os.chdir(PARENT_DIR)
    os.chdir(CURRENT_DIR)
    return trajectories, starting_states


# if __name__ == "__main__":
#     trajectories, starting_states = generate_trajectories_from_files()
#     for i in range(len(trajectories)):
#         env = init_env(render=True)
#         frames = record_trajectory(env, trajectories[i], starting_states[i])
#         clip_name = f'expert_trajectories/expert_trajectory_{i+1}.mp4'
#         generate_clip(frames, clip_name, fps=15)
