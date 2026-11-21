r"""Inspect the trained 2v2 agent's FIRST decision in each lead matchup.

Usage (with the local Showdown server already running):
  .\.venv\Scripts\python.exe -u inspect_2v2_first_turns.py --battles-per-lead 100

Does not train or overwrite weights. This evaluates completed battles (not just
resets), because resetting an unfinished poke-env battle can leave a hanging task.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from prototype import as_tensors, seed_all
from prototype_2v2 import (
    ActorCritic2, MY_TEAMS, OPP_TEAMS, MAX_TURNS,
    bad_matchup, make_env, species,
)


def action_name(action: int, battle) -> str:
    if action < 6:
        team = list(battle.team.values())
        name = species(team[action]) if action < len(team) else 'unknown'
        return f'switch to {name} (action {action})'
    if 6 <= action <= 9:
        moves = list(battle.active_pokemon.moves.values()) if battle.active_pokemon else []
        move_name = moves[action - 6].id if action - 6 < len(moves) else 'unknown'
        return f'{move_name} (action {action})'
    return f'action {action}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', default='results_2v2_10k/policy.pt')
    parser.add_argument('--battles-per-lead', type=int, default=100)
    parser.add_argument('--opponent', choices=['tactical', 'random'], default='tactical')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.battles_per_lead < 1:
        parser.error('--battles-per-lead must be positive')
    seed_all(args.seed)
    model = ActorCritic2()
    model.load_state_dict(torch.load(Path(args.weights), map_location='cpu', weights_only=True))
    model.eval()
    env = make_env(args.opponent)
    try:
        print(f'Model: {args.weights}; opponent: {args.opponent}; {args.battles_per_lead} battles/lead', flush=True)
        for my_idx in range(2):
            for opp_idx in range(2):
                first = Counter()
                wins = legal_to_switch = first_switched = 0
                first_bad = first_bad_switched = 0
                for _ in range(args.battles_per_lead):
                    env.env.agent1.update_team(MY_TEAMS[my_idx])
                    env.env.agent2.update_team(OPP_TEAMS[opp_idx])
                    obs, _ = env.reset()
                    battle = env.env.battle1
                    mask = np.asarray(obs['action_mask'], dtype=bool)
                    can_switch = bool(mask[:6].any())
                    can_move = bool(mask[6:10].any())
                    initial_bad = bad_matchup(battle)
                    with torch.no_grad():
                        inputs, mask_tensor = as_tensors(obs)
                        distribution, _ = model(inputs, mask_tensor)
                        action = int(distribution.probs.argmax(dim=-1).item())
                        # Override ONLY the first decision of the battle.

                        if my_idx == 1 and opp_idx == 0:
                            # Corviknight vs Meowscarada
                            action = 6  # Brave Bird instead of switching

                        elif my_idx == 1 and opp_idx == 1:
                            # Corviknight vs Magnezone
                            action = 1  # Switch to Swampert instead of Brave Bird
                        if not mask[action]:
                            raise ValueError(
                                f"Forced action {action} is illegal in this battle!"
                            )
                    first[action_name(action, battle)] += 1
                    legal_to_switch += int(can_switch and can_move)
                    first_switched += int(action < 6 and can_move)
                    first_bad += int(initial_bad and can_switch and can_move)
                    first_bad_switched += int(initial_bad and can_move and action < 6)
                    terminated = truncated = False
                    while not (terminated or truncated):
                        obs, _, terminated, truncated, _ = env.step(np.int64(action))
                        if terminated or truncated:
                            break
                        if env.env.battle1.turn >= MAX_TURNS:
                            raise RuntimeError(f'Battle still active after {MAX_TURNS} turns; stopping safely.')
                        with torch.no_grad():
                            inputs, mask_tensor = as_tensors(obs)
                            distribution, _ = model(inputs, mask_tensor)
                            action = int(distribution.probs.argmax(dim=-1).item())
                    wins += int(bool(env.env.battle1.won))
                a = ['Swampert', 'Corviknight'][my_idx]
                b = ['Meowscarada', 'Magnezone'][opp_idx]
                print(f'\n{a} vs {b}: {wins}/{args.battles_per_lead} wins', flush=True)
                print(f'  First-turn voluntary switches: {first_switched}/{args.battles_per_lead}', flush=True)
                print(f'  First-turn switch AND attack both legal: {legal_to_switch}/{args.battles_per_lead}', flush=True)
                if first_bad:
                    print(f'  First-turn switches from bad matchup (when legal): {first_bad_switched}/{first_bad}', flush=True)
                for name, n in first.most_common():
                    print(f'  {n:3d} x {name}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
