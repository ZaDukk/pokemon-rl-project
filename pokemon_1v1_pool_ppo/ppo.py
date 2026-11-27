"""Independent PPO implementation: clipped objective, GAE, masked policy, Adam.

Uses PyTorch as a tensor/autograd library, NOT a pretrained or packaged RL agent.
Identical to ../pokemon_3v3_pool_ppo/ppo.py except the class name (ActorCritic3 ->
ActorCritic); the input size comes from the smaller 1v1 encoder in features.py.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from features import N_FEATURES
ACTION_COUNT = 26

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_FEATURES, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.policy = nn.Linear(256, ACTION_COUNT)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if not mask.any(dim=-1).all():
            raise ValueError("No legal actions in action mask")
        z = self.encoder(x)
        logits = self.policy(z).masked_fill(~mask.bool(), -1e9)
        return Categorical(logits=logits), self.value_head(z).squeeze(-1)


def as_tensors(obs: dict):
    x = torch.as_tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
    mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0)
    if x.shape != (1, N_FEATURES) or mask.shape != (1, ACTION_COUNT):
        raise ValueError(f"Unexpected observation/mask shape: {x.shape}, {mask.shape}")
    return x, mask


@torch.no_grad()
def choose_action(model: ActorCritic | None, observation, deterministic=False):
    if model is None:
        choices = np.flatnonzero(observation["action_mask"])
        if len(choices) == 0:
            raise ValueError("No legal actions")
        return int(np.random.choice(choices)), 0.0, 0.0
    state, mask = as_tensors(observation)
    dist, value = model(state, mask)
    action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
    return int(action.item()), float(dist.log_prob(action).item()), float(value.item())


def ppo_update(model, optimizer, batch, bootstrap, *, gamma=.99, gae_lambda=.95,
               clip=.2, epochs=4, minibatch=64, entropy_coef=.02,
               value_coef=.5):
    """Our own clipped PPO and GAE implementation, no third-party RL trainer."""
    rewards = np.asarray(batch["reward"], np.float32)
    values = np.asarray(batch["value"], np.float32)
    dones = np.asarray(batch["done"], np.float32)
    adv = np.zeros_like(rewards)
    gae = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        next_value = bootstrap if i == len(rewards) - 1 else values[i + 1]
        is_live = 1.0 - dones[i]
        delta = rewards[i] + gamma * next_value * is_live - values[i]
        gae = delta + gamma * gae_lambda * is_live * gae
        adv[i] = gae
    returns = adv + values

    obs = torch.as_tensor(np.stack(batch["obs"]), dtype=torch.float32)
    masks = torch.as_tensor(np.stack(batch["mask"]), dtype=torch.bool)
    actions = torch.as_tensor(batch["action"], dtype=torch.long)
    old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32)
    returns = torch.as_tensor(returns, dtype=torch.float32)
    adv = torch.as_tensor(adv, dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    metrics = defaultdict(float)
    batches = 0
    for _ in range(epochs):
        for indices in torch.randperm(len(rewards)).split(minibatch):
            dist, val = model(obs[indices], masks[indices])
            logp = dist.log_prob(actions[indices])
            ratio = (logp - old_logp[indices]).exp()
            policy_loss = -torch.minimum(ratio * adv[indices],
                                         ratio.clamp(1 - clip, 1 + clip) * adv[indices]).mean()
            value_loss = 0.5 * (returns[indices] - val).square().mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.item()
            metrics["approx_kl"] += ((ratio - 1) - (logp - old_logp[indices])).mean().item()
            batches += 1
    return {k: v / max(batches, 1) for k, v in metrics.items()}


