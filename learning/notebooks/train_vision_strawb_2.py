import os
# Set environment variables early, before importing any libraries
os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "/home/emlyn/madrona_mjx/build/cache"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # Limit memory usage

import time
import functools
import jax
import numpy as np
import cv2
from mujoco_playground import manipulation
from mujoco_playground import wrapper
from mujoco_playground._src.manipulation.franka_emika_panda import randomize_vision as randomize

# Configure logging to see potential issues
import logging
logging.basicConfig(level=logging.INFO)

def configure_environment(env_name="PandaPickStrawb", num_envs=4):
    """Configure the environment with proper settings."""
    env_cfg = manipulation.get_default_config(env_name)
    episode_length = int(4 / env_cfg.ctrl_dt)
    
    config_overrides = {
        "episode_length": episode_length,
        "vision": True,
        "obs_noise.brightness": [0.75, 2.0],
        "vision_config.use_rasterizer": True,  # Use rasterizer for faster init
        "vision_config.render_batch_size": num_envs,
        "vision_config.render_width": 84,
        "vision_config.render_height": 84,
        "box_init_range": 0.1,
        "action_history_length": 5,
        "success_threshold": 0.03
    }
    
    print("Creating environment...")
    env = manipulation.load(
        env_name, 
        config=env_cfg,
        config_overrides=config_overrides
    )
    
    randomization_fn = functools.partial(
        randomize.domain_randomize,
        num_worlds=num_envs
    )
    
    env = wrapper.wrap_for_brax_training(
        env,
        vision=True,
        num_vision_envs=num_envs,
        episode_length=episode_length,
        action_repeat=1,
        randomization_fn=randomization_fn
    )
    
    return env, num_envs

def warmup_jax_and_renderer(env, num_envs):
    """Perform a warmup run to initialize JAX and the renderer."""
    print("Starting JAX and renderer warmup...")
    warmup_time = time.time()
    
    # Create a warmup function that performs a reset and a few steps
    @jax.jit
    def warmup_fn(rng):
        reset_state = env.reset(jax.random.split(rng, num_envs))
        # Run a few dummy steps to fully compile everything
        dummy_action = jax.numpy.zeros((num_envs, env.action_size))
        state = reset_state
        for _ in range(2):  # Just a few steps to ensure compilation
            state = env.step(state, dummy_action)
        return state
    
    # Run the warmup with a throwaway key
    warmup_key = jax.random.PRNGKey(42)
    _ = warmup_fn(warmup_key)  # This will be slow
    
    print(f"Warmup complete in {time.time() - warmup_time:.2f} seconds")
    return jax.jit(env.reset), jax.jit(env.step)

def main():
    # 1. Configure the environment
    print("Setting up environment...")
    env, num_envs = configure_environment()
    
    # 2. Perform warmup to compile JAX functions
    jit_reset, jit_step = warmup_jax_and_renderer(env, num_envs)
    
    # 3. Now measure the actual reset time (after compilation)
    print("\nMeasuring actual reset time (post-compilation)...")
    reset_time = time.time()
    state = jit_reset(jax.random.split(jax.random.PRNGKey(0), num_envs))
    actual_reset_time = time.time() - reset_time
    print(f"JIT reset time (after compilation): {actual_reset_time:.4f} seconds")
    
    # 4. Visualize the environment
    if hasattr(state, "obs") and "pixels/view_0" in state.obs:
        print("Displaying image...")
        cv2.imshow("image", state.obs["pixels/view_0"][0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 5. Run a simple demonstration of the environment
    print("\nRunning a simple demonstration...")
    for i in range(10):
        start = time.time()
        # Create a random action
        action = jax.numpy.zeros((num_envs, env.action_size))
        state = jit_step(state, action)
        print(f"Step {i+1} took {time.time() - start:.4f} seconds")
        
        # Display the image
        if i % 3 == 0 and hasattr(state, "obs") and "pixels/view_0" in state.obs:
            cv2.imshow("image", np.array(state.obs["pixels/view_0"][0]))
            cv2.waitKey(1)  # Brief display
    
    cv2.destroyAllWindows()
    print("Demonstration complete!")

if __name__ == "__main__":
    main()