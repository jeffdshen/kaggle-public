from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp as amp
import random
import collections

import gym
from gym.vector import SyncVectorEnv

from .model import Model
from .utils import ArmHelper, ARMS
from .rl_utils import (
    ReplayBuffer,
    get_linear_warmup_power_decay_scheduler,
    make_transitions,
    masked_select,
    multi_argmax,
    multi_max,
    results_to_str,
    set_seed,
    to_torch,
)


def get_model(config):
    return Model(
        box_size=config["box_size"],
        angle_sizes=config["angle_sizes"],
        action_size=config["action_size"],
        stages=config["stages"],
        ff_dim_factor=config["ff_dim_factor"],
        kernel_size=config["kernel_size"],
        activation=config["activation"],
        num_groups=config["num_groups"],
    )


def get_optimizer(model, config):
    return optim.AdamW(
        model.parameters(),
        config["lr"],
        betas=config["betas"],
        eps=config["eps"],
        weight_decay=config["wd"],
    )


def get_epsilon(epsilon, min_epsilon, epsilon_decay, sample_num):
    return max(min_epsilon, epsilon * epsilon_decay**sample_num)


def update_target(target_model, model):
    target_model.load_state_dict(model.state_dict())


def step_episode(
    config, sample_num, train_envs, model, device, prev_obs, replay_buffer
):
    epsilon = get_epsilon(
        config["epsilon"], config["min_epsilon"], config["epsilon_decay"], sample_num
    )
    actions_mask = np.random.rand(config["num_envs"]) < epsilon
    actions = train_envs.action_space.sample()
    with torch.no_grad():
        if np.count_nonzero(~actions_mask) > 0:
            inputs = masked_select(prev_obs, ~actions_mask)
            inputs = to_torch(inputs, device)
            model_actions = model(**inputs)
            actions[~actions_mask] = multi_argmax(
                model_actions, len(config["action_size"])
            )
    obs, rewards, terminated, truncated, infos = train_envs.step(actions)
    transitions = make_transitions(
        prev_obs, obs, actions, rewards, terminated, infos, config["num_envs"]
    )
    for transition in transitions:
        replay_buffer.add(*transition)

    if "_final_info" in infos and infos["_final_info"][0]:
        results = {}
        final_info = infos["final_info"][0]
        results.update({k: final_info[k] for k in ["remaining", "total_cost"]})
        episode_info = final_info["episode"]
        results.update({k: episode_info for k in ["r", "l"]})
        results["sample"] = sample_num
        return results

    return None


def record_env(env_fn, path):
    env = env_fn()
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, video_folder=path)
    return env


def train(config, wandb, save_dir):
    wandb.config.update(config)
    set_seed(config["seed"])
    save_dir = Path(save_dir) / wandb.run.name
    replay_buffer = ReplayBuffer(config["buffer_size"])
    env_fn = lambda: gym.make("Santa2022Game-v0", render_mode="rgb_array")
    record_env_fn = lambda: record_env(env_fn, path=save_dir / "video")

    train_envs = SyncVectorEnv([record_env_fn] + [env_fn] * (config["num_envs"] - 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config)
    target_model = get_model(config)
    model = model.to(device)
    target_model = target_model.to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_linear_warmup_power_decay_scheduler(
        optimizer, config["warmup_steps"], float("inf"), end_multiplier=1.0, power=1.0
    )
    scaler = amp.GradScaler()

    update_target(target_model, model)
    prev_obs, _ = train_envs.reset()
    sample_num = 0
    step_num = 0
    samples_since_eval = 0
    samples_since_target = 0
    results = {}

    def backward(loss):
        nonlocal step_num
        # Backward
        scaler.scale(loss / config["gradient_accumulation"]).backward()

        # Step
        if (step_num + 1) % config["gradient_accumulation"] == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        step_num += 1

    while sample_num < config["num_samples"]:
        results = step_episode(
            config, sample_num, train_envs, model, device, prev_obs, replay_buffer
        )
        if results is not None:
            latest_results = results
        sample_num += config["num_envs"]
        samples_since_eval += config["num_envs"]
        samples_since_target += config["num_envs"]

        for _ in range(0, config["batch_size"], config["model_batch_size"]):
            state, action, reward, next_state, done = replay_buffer.sample(
                config["model_batch_size"]
            )

            ## Actually DDQN (not regular DQN)
            with amp.autocast(enabled=config["autocast"]):
                with torch.no_grad():
                    qt_actions = model(**to_torch(next_state, device))
                    qt_actions = qt_actions.flatten(-len(config["action_size"]))
                    qt_actions = qt_actions.argmax(dim=-1, keepdim=True)
                    qt_values = target_model(**to_torch(next_state, device))
                    qt_values = qt_values.flatten(-len(config["action_size"]))
                    qt_values = qt_values.gather(-1, qt_actions).squeeze(-1)
                    qt_values[done] = 0

                    r = torch.from_numpy(reward).to(device, dtype=torch.float32)
                    qt = r + config["discount_factor"] * qt_values

                q = model(**to_torch(state, device))
                q = q.flatten(-len(config["action_size"]))
                action = np.ravel_multi_index(
                    action.transpose(1, 0), dims=config["action_size"]
                )
                action = qt_actions.new_tensor(action).unsqueeze(-1)
                q = q.gather(-1, action).squeeze(-1)
                loss = F.mse_loss(q, qt)
            backward(loss)

        if samples_since_target >= config["target_update_freq"]:
            samples_since_target -= config["target_update_freq"]
            update_target(target_model, model)

        if samples_since_eval >= config["eval_per_n_samples"]:
            samples_since_eval -= config["eval_per_n_samples"]
            print(results_to_str(latest_results))
            wandb.log(latest_results)
            torch.save(model.state_dict(), save_dir / "latest.pt")
