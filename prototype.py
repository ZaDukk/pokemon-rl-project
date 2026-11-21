"""Small Pokémon Showdown 1v1 RL feasibility experiment.

This file contains an independently written, minimal PPO implementation.
It relies on poke-env for battle simulation and Gymnasium integration ONLY.

First use `python prototype.py smoke --battles 40` to measure battle speed.
Then use `python prototype.py train --steps 10000 --eval-every 2000`.

The format '[Gen 9] RL 1v1' must be installed in the local Showdown server.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import RandomPlayer
from torch import nn
from torch.distributions import Categorical

BATTLE_FORMAT = 'gen9rl1v1'
TEAM = '''Conkeldurr @ Sitrus Berry
Ability: Iron Fist
Level: 100
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Drain Punch
- Mach Punch
- Bulk Up
- Protect'''
N_FEATURES = 7


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def hp(mon) -> float:
    return float(mon.current_hp_fraction) if mon is not None else 0.0


def boost(mon, stat: str) -> float:
    return float(mon.boosts.get(stat, 0)) / 6.0 if mon is not None else 0.0


class OneVsOneEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            name: Box(-1.0, 1.0, shape=(N_FEATURES,), dtype=np.float32)
            for name in self.possible_agents
        }

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        ours = battle.active_pokemon
        theirs = battle.opponent_active_pokemon
        return np.array([
            hp(ours), hp(theirs),
            boost(ours, 'atk'), boost(theirs, 'atk'),
            boost(ours, 'def'), boost(theirs, 'def'),
            min(float(battle.turn) / 40.0, 1.0),
        ], dtype=np.float32)

    def calc_reward(self, battle: AbstractBattle) -> float:
        # A small dense HP signal plus a clear terminal win/loss signal.
        return self.reward_computing_helper(
            battle,
            number_of_pokemons=1,
            hp_value=1.0,
            fainted_value=1.0,
            victory_value=5.0,
        )


def make_env() -> SingleAgentWrapper:
    env = OneVsOneEnv(
        battle_format=BATTLE_FORMAT,
        team=TEAM,
        choose_on_teampreview=False,
        log_level=40,
        open_timeout=None,
        challenge_timeout=45.0,
    )
    # SingleAgentWrapper uses RandomPlayer.choose_move on agent2's battle view.
    # Both environment agents receive the SAME fixed team, hence mirror matches.
    opponent = RandomPlayer(battle_format=BATTLE_FORMAT, start_listening=False)
    return SingleAgentWrapper(env, opponent)


class ActorCritic(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_FEATURES, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.policy = nn.Linear(64, n_actions)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        z = self.encoder(obs)
        logits = self.policy(z).masked_fill(~mask.bool(), -1e9)
        return Categorical(logits=logits), self.value_head(z).squeeze(-1)


def as_tensors(obs: dict):
    x = torch.as_tensor(obs['observation'], dtype=torch.float32).unsqueeze(0)
    mask = torch.as_tensor(obs['action_mask'], dtype=torch.bool).unsqueeze(0)
    if not bool(mask.any()):
        raise RuntimeError('Empty legal action mask. Check the custom format / battle.')
    return x, mask


@torch.no_grad()
def choose_action(model: ActorCritic | None, obs: dict, deterministic=False):
    if model is None:
        return int(np.random.choice(np.flatnonzero(obs['action_mask'])))
    x, mask = as_tensors(obs)
    dist, value = model(x, mask)
    action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
    return int(action.item()), float(dist.log_prob(action).item()), float(value.item())


def evaluate(env: SingleAgentWrapper, model: ActorCritic | None, battles: int):
    """Return win/tie rates and throughput. Do not train during evaluation."""
    wins = ties = decisions = 0
    start = time.perf_counter()
    for _ in range(battles):
        observation, _ = env.reset()
        done = False
        while not done:
            chosen = choose_action(model, observation, deterministic=True)
            action = chosen if model is None else chosen[0]
            observation, _, terminated, truncated, _ = env.step(np.int64(action))
            decisions += 1
            done = bool(terminated or truncated)
        battle = env.env.battle1
        wins += int(bool(battle.won))
        ties += int(not battle.won and not battle.lost)
    seconds = time.perf_counter() - start
    return dict(
        win_rate=wins / battles,
        tie_rate=ties / battles,
        battles_per_second=battles / max(seconds, 1e-9),
        decisions_per_second=decisions / max(seconds, 1e-9),
        mean_decisions=decisions / battles,
        elapsed_seconds=seconds,
    )


def ppo_update(model, optimizer, batch, bootstrap_value, gamma=.99, gae_lambda=.95,
               clip=.2, epochs=4, minibatch=64, ent_coef=.02, value_coef=.5):
    # Store the value at the next observation at a rollout boundary.
    rewards = np.array(batch['reward'], dtype=np.float32)
    values = np.array(batch['value'], dtype=np.float32)
    dones = np.array(batch['done'], dtype=np.float32)
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    for i in reversed(range(len(rewards))):
        if i == len(rewards)-1:
            next_value = bootstrap_value
        else:
            next_value = values[i+1]
        nonterminal = 1.0 - dones[i]
        delta = rewards[i] + gamma * next_value * nonterminal - values[i]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[i] = last_gae
    returns = advantages + values

    obs = torch.tensor(np.stack(batch['obs']), dtype=torch.float32)
    masks = torch.tensor(np.stack(batch['mask']), dtype=torch.bool)
    actions = torch.tensor(batch['action'], dtype=torch.long)
    old_logp = torch.tensor(batch['logp'], dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    n = len(rewards)
    for _ in range(epochs):
        for ids in torch.randperm(n).split(minibatch):
            dist, predictions = model(obs[ids], masks[ids])
            logp = dist.log_prob(actions[ids])
            ratio = (logp - old_logp[ids]).exp()
            a = advantages[ids]
            loss_policy = -torch.minimum(ratio * a, ratio.clamp(1-clip, 1+clip) * a).mean()
            loss_value = 0.5 * (returns[ids] - predictions).square().mean()
            loss = loss_policy + value_coef * loss_value - ent_coef * dist.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


def train(args):
    model = ActorCritic(n_actions=26)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_env = make_env()
    eval_env = make_env()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    csv_file = output / 'results.csv'
    try:
        baseline = evaluate(eval_env, None, args.eval_battles)
        print(f'Untrained random-policy baseline: {baseline["win_rate"]:.1%} wins; '
              f'{baseline["battles_per_second"]:.1f} battles/s', flush=True)
        with csv_file.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                'training_decisions', 'training_battles', 'evaluation_win_rate',
                'evaluation_tie_rate', 'evaluation_battles_per_second',
                'training_decisions_per_second', 'training_seconds',
            ])
            writer.writeheader()
            writer.writerow(dict(
                training_decisions=0, training_battles=0,
                evaluation_win_rate=baseline['win_rate'],
                evaluation_tie_rate=baseline['tie_rate'],
                evaluation_battles_per_second=baseline['battles_per_second'],
                training_decisions_per_second=0, training_seconds=0,
            ))

        obs, _ = train_env.reset()
        total_steps = 0
        total_battles = 0
        total_training_time = 0.0
        last_report = 0
        while total_steps < args.steps:
            batch = {k: [] for k in ('obs', 'mask', 'action', 'logp', 'value', 'reward', 'done')}
            start = time.perf_counter()
            for _ in range(min(args.rollout, args.steps-total_steps)):
                action, logp, value = choose_action(model, obs)
                batch['obs'].append(obs['observation'].copy())
                batch['mask'].append(obs['action_mask'].copy())
                batch['action'].append(action)
                batch['logp'].append(logp)
                batch['value'].append(value)
                obs, reward, terminated, truncated, _ = train_env.step(np.int64(action))
                done = bool(terminated or truncated)
                batch['reward'].append(float(reward))
                batch['done'].append(done)
                total_steps += 1
                if done:
                    total_battles += 1
                    obs, _ = train_env.reset()
            if batch['done'][-1]:
                bootstrap = 0.0
            else:
                with torch.no_grad():
                    x, mask = as_tensors(obs)
                    _, v = model(x, mask)
                    bootstrap = float(v.item())
            ppo_update(model, optimizer, batch, bootstrap)
            total_training_time += time.perf_counter() - start

            if total_steps-last_report >= args.eval_every or total_steps == args.steps:
                stats = evaluate(eval_env, model, args.eval_battles)
                speed = total_steps / max(total_training_time, 1e-9)
                print(f'{total_steps:>6} train decisions, {total_battles:>5} battles | '
                      f'eval wins {stats["win_rate"]:.1%} | '
                      f'train {speed:.1f} decisions/s', flush=True)
                with csv_file.open('a', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow([
                        total_steps, total_battles, stats['win_rate'],
                        stats['tie_rate'], stats['battles_per_second'],
                        speed, total_training_time,
                    ])
                torch.save(model.state_dict(), output / 'policy.pt')
                last_report = total_steps
        print(f'Results saved to {csv_file}; weights saved to {output / "policy.pt"}')
    finally:
        train_env.close()
        eval_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    smoke = sub.add_parser('smoke', help='Run random vs random and benchmark speed')
    smoke.add_argument('--battles', type=int, default=40)
    smoke.add_argument('--seed', type=int, default=42)
    training = sub.add_parser('train', help='Train self-implemented masked PPO')
    training.add_argument('--steps', type=int, default=10000)
    training.add_argument('--rollout', type=int, default=256)
    training.add_argument('--eval-every', type=int, default=2000)
    training.add_argument('--eval-battles', type=int, default=100)
    training.add_argument('--lr', type=float, default=3e-4)
    training.add_argument('--output', default='results')
    training.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.mode == 'smoke':
        seed_all(args.seed)
        env = make_env()
        try:
            stats = evaluate(env, None, args.battles)
            for key, value in stats.items():
                print(f'{key}: {value:.3f}')
        finally:
            env.close()
    else:
        seed_all(args.seed)
        train(args)


if __name__ == '__main__':
    main()
