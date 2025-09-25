import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# train_rl_ot2_local.py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from ot2_gym_wrapper import OT2Env
import os

# HYPERPARAMETERS 
learning_rate = 0.0003
batch_size = 64
n_steps = 2048
n_epochs = 10
total_timesteps = 1_000_000

# OUTPUT DIR 
model_dir = "models/local_run"
os.makedirs(model_dir, exist_ok=True)

# ENV 
env = OT2Env()

# MODEL 
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=learning_rate,
    batch_size=batch_size,
    n_steps=n_steps,
    n_epochs=n_epochs,
    verbose=1,
    tensorboard_log="runs/local_tensorboard"
)

# CHECKPOINTING 
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=model_dir,
    name_prefix="ppo_checkpoint"
)

# TRAIN 
model.learn(
    total_timesteps=total_timesteps,
    callback=checkpoint_callback
)

# SAVE FINAL MODEL 
os.makedirs("models", exist_ok=True)
model.save("models/model_v1.zip")
print(f"Training complete. Final model saved to {model_dir}/ppo_final")
