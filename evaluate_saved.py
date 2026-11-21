r"""Evaluate an already-trained Pokemon 1v1 PPO policy without retraining.

Place this file alongside prototype.py and results/policy.pt, then run:
    .\.venv\Scripts\python.exe evaluate_saved.py --battles 300

It tests random and maximum-base-power opponents, and reports which action
indices the learned policy selects. No training weights are modified.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.player import RandomPlayer, MaxBasePowerPlayer

from prototype import (
    ActorCritic,
    BATTLE_FORMAT,
    TEAM,
    OneVsOneEnv,
    choose_action,
    seed_all,
)


def play(model, opponent_class, battles):
    opponent = opponent_class(battle_format=BATTLE_FORMAT, start_listening=False)
    raw_env = OneVsOneEnv(
        battle_format=BATTLE_FORMAT,
        team=TEAM,
        choose_on_teampreview=False,
        log_level=40,
        open_timeout=None,
        challenge_timeout=45.0,
    )
    env = SingleAgentWrapper(raw_env, opponent)
    wins = ties = steps = 0
    actions = Counter()
    try:
        for _ in range(battles):
            observation, _ = env.reset()
            done = False
            while not done:
                action, _, _ = choose_action(model, observation, deterministic=True)
                actions[action] += 1
                observation, _, terminated, truncated, _ = env.step(np.int64(action))
                steps += 1
                done = bool(terminated or truncated)
            result = env.env.battle1
            wins += int(bool(result.won))
            ties += int(not result.won and not result.lost)
    finally:
        env.close()
    print(f'{opponent_class.__name__}: {wins}/{battles} wins = {wins/battles:.1%}, '
          f'{ties} ties, {steps/battles:.1f} decisions/battle')
    print('  Selected action indices:', dict(sorted(actions.items())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--battles', type=int, default=300)
    parser.add_argument('--weights', default='results/policy.pt')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()
    if args.battles < 1:
        parser.error('--battles must be >= 1')
    seed_all(args.seed)
    model = ActorCritic(n_actions=26)
    weights_path = Path(args.weights)
    if not weights_path.is_file():
        parser.error(f'Weights file does not exist: {weights_path}')
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    model.eval()
    print(f'Evaluating saved weights: {weights_path}\n')
    for opponent_class in (RandomPlayer, MaxBasePowerPlayer):
        play(model, opponent_class, args.battles)


if __name__ == '__main__':
    main()
