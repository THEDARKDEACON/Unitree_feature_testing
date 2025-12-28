from stable_baselines3 import PPO
from envs.h1_env import H1Env
import os
import glob

def main():
    # 1. Find latest checkpoint
    checkpoint_dir = "./h1_checkpoints/"
    if not os.path.exists(checkpoint_dir):
        print("No checkpoints found. Run train_ppo.py first.")
        return

    list_of_files = glob.glob(os.path.join(checkpoint_dir, '*.zip')) 
    if not list_of_files:
        print("No .zip files in checkpoint directory.")
        return
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading latest checkpoint: {latest_file}")
    
    # 2. Setup Monitor
    env = H1Env(render_mode="human")
    
    # 3. Load Model
    model = PPO.load(latest_file)
    
    # 4. Play
    print("Running simulation... Press Ctrl+C to stop.")
    obs, info = env.reset()
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("Closing viewer.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
