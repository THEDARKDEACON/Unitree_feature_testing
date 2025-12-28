from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.h1_env import H1Env
import os

def main():
    # 1. Create Environment
    env = H1Env()
    
    # 2. Check Environment
    print("Checking environment...")
    check_env(env)
    print("Environment verified.")
    
    # 3. Define Callbacks
    from stable_baselines3.common.callbacks import CheckpointCallback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./h1_checkpoints/",
        name_prefix="h1_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # 4. Define PPO Model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log="./h1_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )
    
    # 5. Train
    print("Starting training with checkpoints...")
    try:
        model.learn(total_timesteps=1_000_000, callback=checkpoint_callback) 
    except KeyboardInterrupt:
        print("Training interrupted.")
        
    # 6. Save Final
    print("Saving model...")
    model.save("h1_ppo_walk_final")
    print("Model saved.")

if __name__ == "__main__":
    main()
