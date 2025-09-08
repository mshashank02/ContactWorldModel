import argparse
import json
from copy import deepcopy
from functools import partial
import gymnasium as gym
import gymnasium_robotics
import os
import numpy as np
from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, VecNormalize, DummyVecEnv
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from gymnasium.wrappers import TimeLimit
import wandb
from wandb.integration.sb3 import WandbCallback
import warnings
from custom_envs.hand_block_forward_face_env import MujocoHandBlockForwardFaceTouchEnv
from custom_envs.hand_block_yaw import MujocoHandBlockYawTouchEnv
from custom_envs.dynamic_touch_env import DynamicXMLTouchEnv
from custom_wrappers.remove_object_state import RemoveObjectStateWrapper



# ignore warning. it does not affect the training
warnings.filterwarnings("ignore", message=".*method is not within the observation space*")

ENV_HYPERPARAMS = {
    "n_timesteps": 16e6,
    "policy": "MultiInputPolicy",
    "buffer_size": 1000000,
    "ent_coef": "auto",
    "batch_size": 2048,
    "gamma": 0.95,
    "learning_rate": 1e-3,
    "learning_starts": 4000,
    "tau": 0.05,
    "n_sampled_goal": 4,
    "goal_selection_strategy": "future",
    "arch": [512, 512, 512],
    "n_critics": 2,
}

def parse_args():
    parser = argparse.ArgumentParser()
    # Experiment config
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--verbose", type=int, default=2,
            help="the verbosity of the logs")
    parser.add_argument("--num-envs", type=int, default=16,
        help="number of parallel environments")
    parser.add_argument("--eval-freq", type=int, default=10000,
        help="frequency of evaluation (in timesteps)")
    parser.add_argument("--eval-episodes", type=int, default=50,
        help="number of episodes for evaluation")
    parser.add_argument("--save-freq", type=int, default=50000,
        help="frequency of saving model and stats (in timesteps)")
    parser.add_argument("--gradient-save-freq", type=int, default=100000,
        help="frequency of saving gradients (in timesteps)")
    parser.add_argument("--model-save-freq", type=int, default=100000,
        help="frequency of saving model (in timesteps)")
    
    # Environment
    parser.add_argument("--env-id", type=str, default="HandManipulateBlockRotateXYZ-v1",
        help="env id")
    parser.add_argument("--xml-path", type=str, required=True,
                        help="Path to generated MuJoCo XML")
    
    # model hyperparameters
    parser.add_argument("--n-timesteps", type=float, default=ENV_HYPERPARAMS["n_timesteps"],
        help="total number of timesteps")
    parser.add_argument("--buffer-size", type=int, default=ENV_HYPERPARAMS["buffer_size"],
        help="replay buffer size")
    parser.add_argument("--batch-size", type=int, default=ENV_HYPERPARAMS["batch_size"],
        help="batch size")
    parser.add_argument("--gamma", type=float, default=ENV_HYPERPARAMS["gamma"],
        help="discount factor")
    parser.add_argument("--learning-rate", type=float, default=ENV_HYPERPARAMS["learning_rate"],
        help="learning rate")
    parser.add_argument("--learning-starts", type=int, default=ENV_HYPERPARAMS["learning_starts"],
        help="steps before learning starts")
    parser.add_argument("--tau", type=float, default=ENV_HYPERPARAMS["tau"],
        help="tau")
    parser.add_argument("--ent-coef", type=str, default=ENV_HYPERPARAMS["ent_coef"],
        help="entropy coefficient")
    
    # HER hyperparameters
    parser.add_argument("--n-sampled-goal", type=int, default=ENV_HYPERPARAMS["n_sampled_goal"],
        help="number of sampled goals for HER")
    parser.add_argument("--goal-selection-strategy", type=str, default=ENV_HYPERPARAMS["goal_selection_strategy"],
        help="goal selection strategy for HER")
    parser.add_argument("--arch", type=int, nargs="+", default=ENV_HYPERPARAMS["arch"],
        help="network architecture (list of layer sizes)")
    parser.add_argument("--n-critics", type=int, default=ENV_HYPERPARAMS["n_critics"],
        help="number of critic networks")
    

    # WandB config
    parser.add_argument("--wandb-project", type=str, default="in-hand manipulation",
        help="wandb project name")
    parser.add_argument("--wandb-name", type=str, default=None,
        help="wandb run name")
    
    #GP-BO configs
    parser.add_argument("--metrics-json", type=str, default=None)
    parser.add_argument("--task-name", type=str, default=None)
    args = parser.parse_args()

    # Normalize to absolute path and sanity-check
    args.xml_path = os.path.abspath(args.xml_path)
    if not os.path.isfile(args.xml_path):
        raise FileNotFoundError(f"XML not found at {args.xml_path}")
    
    # Auto-generate wandb name if not provided
    if args.wandb_name is None:
        args.wandb_name = f"{args.env_id}_{args.num_envs}env_{args.seed}"
    
    return args

def make_env(xml_path, seed, rank, max_steps = 100):
    def _init():
        env = DynamicXMLTouchEnv(xml_path=xml_path, render_mode=None)
        env = TimeLimit(env, max_episode_steps=max_steps)
        env.reset(seed=seed + rank)
        env = Monitor(env)
        env = TimeFeatureWrapper(env)
        return env
    return _init

def make_eval_env(xml_path, seed, max_steps = 100):
    def _init():
        env = DynamicXMLTouchEnv(xml_path=xml_path, render_mode="rgb_array")
        env = TimeLimit(env, max_episode_steps=max_steps)
        env.reset(seed=seed)
        env = Monitor(env) 
        env = TimeFeatureWrapper(env)
        return env
    
    env = DummyVecEnv([_init])
    env.seed(seed + 1000)
    return env

def evaluate_policy(model, env, n_eval_episodes=10):
    episode_rewards = []
    episode_successes = []
    
    for _ in range(n_eval_episodes):
        reset_return = env.reset()
        
        if isinstance(reset_return, tuple):
            if len(reset_return) >= 1:
                obs = reset_return[0]
            else:
                print("Warning: reset() returned an empty tuple")
                continue
        elif isinstance(reset_return, dict):
            obs = reset_return
        else:
            obs = reset_return
        
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_return = env.step(action)
            
            if isinstance(step_return, tuple):
                if len(step_return) == 5:  # obs, reward, terminated, truncated, info
                    obs, reward, terminated, truncated, info = step_return
                    
                    # Handle different formats of terminated/truncated
                    if isinstance(terminated, bool):
                        done = terminated or truncated
                    elif hasattr(terminated, '__getitem__'):
                        done = terminated[0] or truncated[0]
                    else:
                        done = bool(terminated) or bool(truncated)
                        
                    if hasattr(reward, '__getitem__'):
                        episode_reward += reward[0]
                    else:
                        episode_reward += reward
                        
                    # Check for success
                    if done and isinstance(info, dict) and 'is_success' in info:
                        episode_successes.append(float(info['is_success']))
                    elif done and hasattr(info, '__getitem__') and len(info) > 0:
                        if isinstance(info[0], dict) and 'is_success' in info[0]:
                            episode_successes.append(float(info[0]['is_success']))
                
                elif len(step_return) == 4:  # obs, reward, done, info 
                    obs, reward, done_var, info = step_return
                    
                    if isinstance(done_var, bool):
                        done = done_var
                    elif hasattr(done_var, '__getitem__'):
                        done = done_var[0]
                    else:
                        done = bool(done_var)
                    
                    if hasattr(reward, '__getitem__'):
                        episode_reward += reward[0]
                    else:
                        episode_reward += reward
                        
                    # Check for success
                    if done and isinstance(info, dict) and 'is_success' in info:
                        episode_successes.append(float(info['is_success']))
                    elif done and hasattr(info, '__getitem__') and len(info) > 0:
                        if isinstance(info[0], dict) and 'is_success' in info[0]:
                            episode_successes.append(float(info[0]['is_success']))
            else:
                print(f"Warning: Unexpected return format from step(): {type(step_return)}")
                done = True
        
        episode_rewards.append(episode_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    # Calculate success rate
    success_rate = np.mean(episode_successes) if episode_successes else None
    
    eval_metrics = {
        'eval/mean_reward': mean_reward,
        'eval/std_reward': std_reward,
    }
    
    if success_rate is not None:
        eval_metrics['eval/success_rate'] = success_rate
    
    return eval_metrics

if __name__ == "__main__":
    args = parse_args()
    print(f"Starting training with environment: {args.env_id}")
    print(f"Using {args.num_envs} parallel environments")
    print(f"Eval frequency: {args.eval_freq} timesteps")
    print(f"Total timesteps: {args.n_timesteps}")
    
    # Ensure directories exist
    os.makedirs(f"videos/{args.env_id}_{args.seed}", exist_ok=True)
    os.makedirs(f"models/{args.env_id}", exist_ok=True)
    
    # Build hyperparameters dict
    hyperparams = {
        "policy": "MultiInputPolicy",
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "learning_starts": args.learning_starts,
        "tau": args.tau,
        "ent_coef": args.ent_coef,
        "replay_buffer_kwargs": {
            "goal_selection_strategy": args.goal_selection_strategy,
            "n_sampled_goal": args.n_sampled_goal,
        },
        "policy_kwargs": {
            "net_arch": args.arch,
            "n_critics": args.n_critics,
        }
    }
    
    if hyperparams["ent_coef"] != "auto":
        hyperparams["ent_coef"] = float(hyperparams["ent_coef"])
    

    env_config = {
        'env_id': args.env_id,
        'num_envs': args.num_envs,
        'eval_freq': args.eval_freq,
        'eval_episodes': args.eval_episodes,
        'save_freq': args.save_freq,
        **hyperparams
    }
    
    run = wandb.init(
        project=args.wandb_project,
        config=env_config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=args.wandb_name
    )
    
    # Create parallel environments
    xml_path = args.xml_path  # passed from generate_and_train
    env = SubprocVecEnv([
        make_env(args.xml_path, args.seed, i)
        for i in range(args.num_envs)
    ])

    normalize_kwargs = {"gamma": hyperparams["gamma"]}
    env = VecNormalize(env, **normalize_kwargs)

    # env = VecVideoRecorder(
    #     env,
    #     f"videos/{args.env_id}_{args.seed}",
    #     record_video_trigger=lambda x: x % args.model_save_freq == 0,
    #     video_length=200
    # )

    # Eval env (also direct)
    eval_env = make_eval_env(args.xml_path, args.seed)
    eval_env = VecNormalize(eval_env, **normalize_kwargs)
    eval_env.training = False
    eval_env.norm_reward = False
    
    # Adjust buffer size for multiple environments
    if args.num_envs > 1:
        hyperparams["buffer_size"] = max(hyperparams["buffer_size"], hyperparams["buffer_size"] * args.num_envs // 4)
        print(f"Adjusted buffer size: {hyperparams['buffer_size']}")
    
    n_timesteps = int(args.n_timesteps)
    
    # Create model
    model = TQC(
        env=env, 
        replay_buffer_class=HerReplayBuffer, 
        verbose=args.verbose,
        seed=args.seed, 
        device='cuda', 
        tensorboard_log=f"runs/{args.env_id}_{args.num_envs}env_{args.seed}", 
        **hyperparams
    )

    class EvalAndSaveCallback(WandbCallback):
        def __init__(self, vec_env, eval_env, model, save_freq, eval_freq, eval_episodes, 
                     save_path, env_id, seed, normalize_kwargs,xml_path, metrics_path=None, total_timesteps=None, task_label=None, **kwargs):
            super().__init__(**kwargs)
            self.vec_env = vec_env
            self.eval_env = eval_env
            self.model = model
            self.save_freq = save_freq
            self.eval_freq = eval_freq
            self.eval_episodes = eval_episodes
            self.save_path = save_path
            self.env_id = env_id
            self.seed = seed
            self.normalize_kwargs = normalize_kwargs
            self.xml_path = xml_path
            self.best_success_rate = 0.0

            #New BO metrics 
            self.metrics_path = metrics_path
            self.total_timesteps = int(total_timesteps) if total_timesteps is not None else None
            self.task_label = (task_label or env_id)
            self.checkpoint_steps = []
            self.success_curve = []
            self._last_eval_ts = 0
            self._last_save_ts = 0


            
        def _on_step(self):
            super()._on_step()
            step_ts = int(self.model.num_timesteps)
            
            if step_ts - self._last_save_ts >= self.save_freq:
                stats_path = os.path.join(self.save_path, f"vecnorm_{self.n_calls}.pkl")
                self.vec_env.save(stats_path)
                wandb.save(stats_path)
                print(f"Saved VecNormalize stats to {stats_path}")
                
                # Save the model
                model_path = os.path.join(self.save_path, f"model_{self.n_calls}_steps.zip")
                self.model.save(model_path)
                wandb.save(model_path)
                print(f"Saved model to {model_path}")

                self._last_save_ts = step_ts
            
            # Run evaluation periodically
            if step_ts - self._last_eval_ts >= self.eval_freq:
                print(f"\nRunning evaluation at {step_ts} timesteps...")
                
                try:
                    self.eval_env.obs_rms = deepcopy(self.vec_env.obs_rms)
                    self.eval_env.ret_rms = deepcopy(self.vec_env.ret_rms)
                    print("Successfully copied normalization stats")
                except Exception as e:
                    print(f"Warning: Could not copy normalization stats: {e}")
                
                # Run evaluation and log metrics
                eval_metrics = evaluate_policy(
                    model=self.model, 
                    env=self.eval_env, 
                    n_eval_episodes=self.eval_episodes
                )
                
                # Log evaluation metrics to wandb
                wandb.log(eval_metrics, step=self.n_calls)
                
                print(f"Evaluation results: {eval_metrics}")

                #Record success curve point 
                succ = float(eval_metrics.get('eval/success_rate', 0.0))
                self.checkpoint_steps.append(step_ts)
                self.success_curve.append(succ)
                
                # Save best model based on success rate if available
                if 'eval/success_rate' in eval_metrics and eval_metrics['eval/success_rate'] > self.best_success_rate:
                    self.best_success_rate = eval_metrics['eval/success_rate']
                    best_model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}_steps.zip")
                    self.model.save(best_model_path)
                    wandb.save(best_model_path)
                    print(f"New best model with success rate {self.best_success_rate:.2f} saved to {best_model_path}")
                
                try:
                    print(f"Creating evaluation video at step {self.n_calls}...")
                    
                    video_path = f"videos/{self.env_id}_{self.seed}/eval_{self.n_calls}"
                    os.makedirs(video_path, exist_ok=True)
                    
                    video_eval_env = make_eval_env(self.xml_path, self.seed)
                    video_eval_env = VecNormalize(video_eval_env, **self.normalize_kwargs)
                    
                    video_eval_env.obs_rms = deepcopy(self.eval_env.obs_rms)
                    video_eval_env.ret_rms = deepcopy(self.eval_env.ret_rms)
                    

                    video_eval_env.training = False
                    video_eval_env.norm_reward = False
                    
                    video_env = VecVideoRecorder(
                        video_eval_env,
                        video_path,
                        record_video_trigger=lambda x: x == 0,  # record only first episode
                        video_length=200,
                        name_prefix=f"eval-{self.n_calls}"
                    )
                    
                    reset_return = video_env.reset()
                    if isinstance(reset_return, tuple):
                        if len(reset_return) >= 1:
                            obs = reset_return[0]
                        else:
                            raise ValueError("Reset returned an empty tuple")
                    elif isinstance(reset_return, dict):
                        obs = reset_return
                    else:
                        obs = reset_return
                    
                    try:
                        video_env.render("rgb_array")
                    except Exception as e:
                        print(f"Initial render failed, but continuing: {e}")
                        
                    done = False
                    step_count = 0
                    max_steps = 200  # Maximum video length
                    
                    print("Recording evaluation episode...")
                    while not done and step_count < max_steps:
                        action, _ = self.model.predict(obs, deterministic=True)
                        
                        try:
                            video_env.render("rgb_array")
                        except Exception as e:
                            pass 
                        
                        step_return = video_env.step(action)
                        
                        if isinstance(step_return, tuple):
                            if len(step_return) == 5:  # obs, reward, terminated, truncated, info
                                obs, _, terminated, truncated, _ = step_return
                                
                                if isinstance(terminated, bool):
                                    done = terminated or truncated
                                elif hasattr(terminated, '__getitem__'):
                                    done = terminated[0] or truncated[0]
                                else:
                                    done = bool(terminated) or bool(truncated)
                            
                            elif len(step_return) == 4:  # obs, reward, done, info (old Gym API)
                                obs, _, done_var, _ = step_return
                                
                                if isinstance(done_var, bool):
                                    done = done_var
                                elif hasattr(done_var, '__getitem__'):
                                    done = done_var[0]
                                else:
                                    done = bool(done_var)
                        else:
                            print(f"Warning: Unexpected return format from step(): {type(step_return)}")
                            done = True
                        
                        try:
                            video_env.render("rgb_array")
                        except Exception as e:
                            pass  # VecVideoRecorder might handle this internally
                            
                        step_count += 1
                    
                    video_env.close()
                    print(f"Finished recording evaluation video after {step_count} steps")
                
                    video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
                    if video_files:
                        eval_video_path = os.path.join(video_path, video_files[0])
                        print(f"Found video file: {eval_video_path}")
                        
                        if os.path.exists(eval_video_path) and os.path.getsize(eval_video_path) > 1000:
                            print(f"Logging video to wandb: {eval_video_path}")
                            wandb.log({
                                "eval/video": wandb.Video(eval_video_path, fps=30, format="mp4"),
                                "eval/video_step": self.n_calls
                            }, step=self.n_calls)
                            print("Successfully logged video to wandb")
                        else:
                            print(f"Warning: Video file is empty or too small: {eval_video_path}")
                    else:
                        print(f"Warning: No video files found in {video_path}")
                        
                except Exception as e:
                    print(f"Error creating evaluation video: {e}")
                    import traceback
                    traceback.print_exc()
                self._last_eval_ts = step_ts

            return True
        
        def _on_training_end(self) -> None:
            #ensure at least one eval point 
            step_ts = int(self.model.num_timesteps)
            if not self.checkpoint_steps or self.checkpoint_steps[-1] < step_ts:
                eval_metrics = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.eval_episodes)
                succ = float(eval_metrics.get('eval/success_rate', 0.0))
                self.checkpoint_steps.append(step_ts)
                self.success_curve.append(succ)
                wandb.log(eval_metrics, step=step_ts)
            
            # write BO metrics JSON
            if self.metrics_path:
                denom = self.total_timesteps if self.total_timesteps else max(1, step_ts)
                fracs = [min(1.0, s / denom) for s in self.checkpoint_steps]  # 0..1 axis
                task = self.task_label
                data = {
                    "tasks": [task],
                    "checkpoints": fracs,
                    "success": {task: [float(x) for x in self.success_curve]},
                    "final_success": {task: float(self.success_curve[-1])}
                }
                os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
                with open(self.metrics_path, "w") as f:
                    json.dump(data, f, indent=2)

            # print a simple scalar for fallback parsers
            print(f"FINAL_SCORE: {self.success_curve[-1]:.6f}")
    
    # Custom callback
    model.learn(
        total_timesteps=n_timesteps,
        callback=EvalAndSaveCallback(
            vec_env=env,
            xml_path=args.xml_path,
            eval_env=eval_env,
            model=model,
            save_freq=args.save_freq,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            save_path=f"models/{args.env_id}",
            env_id=args.env_id,
            seed=args.seed,
            normalize_kwargs=normalize_kwargs,
            #for GP-BO 
            metrics_path=args.metrics_json,
            total_timesteps=n_timesteps,
            task_label=(args.task_name or args.env_id),


            gradient_save_freq=args.gradient_save_freq,
            model_save_freq=args.model_save_freq,
            model_save_path=f"models/{args.env_id}",
            verbose=args.verbose,
        ),
    )
    
    # Save final model and normalization stats
    final_model_path = f"models/{args.env_id}/{args.env_id}_{args.num_envs}env_{args.seed}_final"
    final_stats_path = f"models/{args.env_id}/{args.env_id}_{args.num_envs}env_{args.seed}_vecnorm_final.pkl"
    
    model.save(final_model_path)
    env.save(final_stats_path)
    
    # Run a final evaluation
    final_eval_metrics = evaluate_policy(model, eval_env, n_eval_episodes=50)
    wandb.log(final_eval_metrics, step=n_timesteps)
    
    print(f"Training finished!")
    print(f"Final evaluation results: {final_eval_metrics}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Final normalization stats saved to: {final_stats_path}")
    
    run.finish()