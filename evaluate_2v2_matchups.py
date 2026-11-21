"""Diagnose a saved 2v2 PPO agent by lead matchup and opponent policy.

Run from the same directory as prototype.py and prototype_2v2.py, with the
local Pokémon Showdown server running and the custom format installed.

Example:
    python evaluate_2v2_matchups.py --weights results_2v2_10k/policy.pt --battles-per-lead 100

This does not retrain or modify the model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from prototype import choose_action, seed_all
from prototype_2v2 import (
    ActorCritic2,
    MY_TEAMS,
    OPP_TEAMS,
    MAX_TURNS,
    bad_matchup,
    make_env,
    species,
)


def run_condition(env, model, *, my_team_index, opponent_team_index, n):
    wins = ties = decisions = voluntary = advantageous = bad_opportunities = 0
    bad_switches = 0
    voluntary_by_action = {}

    for _ in range(n):
        # Start each test with the exact pair of leads shown in the summary.
        env.env.agent1.update_team(MY_TEAMS[my_team_index])
        env.env.agent2.update_team(OPP_TEAMS[opponent_team_index])
        observation, _ = env.reset()
        finished = False
        while not finished:
            battle = env.env.battle1
            mask = observation['action_mask']
            can_move = bool(np.any(mask[6:10]))
            can_switch = bool(np.any(mask[:6]))
            bad = bad_matchup(battle)
            if bad and can_move and can_switch:
                bad_opportunities += 1

            if model is None:
                action = int(np.random.choice(np.flatnonzero(mask)))
            else:
                action, _, _ = choose_action(model, observation, deterministic=True)
            if action < 6 and can_move:
                voluntary += 1
                voluntary_by_action[action] = voluntary_by_action.get(action, 0) + 1
                if bad:
                    bad_switches += 1
                else:
                    advantageous += 1

            observation, _, terminated, truncated, _ = env.step(np.int64(action))
            decisions += 1
            finished = bool(terminated or truncated)
            if not finished and env.env.battle1.turn >= MAX_TURNS:
                raise RuntimeError(
                    f"Battle reached {MAX_TURNS} turns without ending. "
                    "Stopping instead of resetting an unfinished Showdown battle. "
                    "Fix the prototype's turn cap before larger experiments."
                )
        outcome = env.env.battle1
        wins += int(bool(outcome.won))
        ties += int(not outcome.won and not outcome.lost)

    return {
        'wins': wins,
        'ties': ties,
        'win_rate': wins / n,
        'decisions_per_battle': decisions / n,
        'voluntary_per_battle': voluntary / n,
        'bad_opportunities': bad_opportunities,
        'bad_switches': bad_switches,
        'bad_switch_rate': (bad_switches / bad_opportunities
                            if bad_opportunities else float('nan')),
        'other_voluntary_switches': advantageous,
        'switch_action_counts': voluntary_by_action,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', default='results_2v2_10k/policy.pt')
    parser.add_argument('--battles-per-lead', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--only', choices=['random', 'tactical', 'both'], default='both')
    args = parser.parse_args()
    if args.battles_per_lead <= 0:
        parser.error('--battles-per-lead must be positive')
    seed_all(args.seed)
    model = ActorCritic2()
    try:
        state = torch.load(Path(args.weights), map_location='cpu', weights_only=True)
        model.load_state_dict(state)
    except Exception as exc:
        raise RuntimeError(f'Cannot load saved 2v2 checkpoint at {args.weights!r}: {exc}') from exc
    model.eval()

    regimes = ['random', 'tactical'] if args.only == 'both' else [args.only]
    own_leads = ['Swampert', 'Corviknight']
    other_leads = ['Meowscarada', 'Magnezone']

    print(f'Loaded {args.weights}. Evaluating {args.battles_per_lead} battles per lead per opponent.', flush=True)
    for regime in regimes:
        env = make_env(regime)
        totals = {'wins': 0, 'ties': 0, 'n': 0}
        print(f'\nOpponent policy: {regime}', flush=True)
        print('Lead matchup                       | Wins      | Switches/battle | Bad-matchup switch rate')
        try:
            for own in range(2):
                for foe in range(2):
                    result = run_condition(env, model, my_team_index=own,
                                           opponent_team_index=foe,
                                           n=args.battles_per_lead)
                    totals['wins'] += result['wins']
                    totals['ties'] += result['ties']
                    totals['n'] += args.battles_per_lead
                    name = f'{own_leads[own]} vs {other_leads[foe]}'
                    bad_rate = result['bad_switch_rate']
                    bad_text = f'{bad_rate:5.1%} ({result["bad_switches"]}/{result["bad_opportunities"]})' if np.isfinite(bad_rate) else 'N/A (no opportunities)'
                    print(f'{name:34} | {result["wins"]:3}/{args.battles_per_lead:<3} | '
                          f'{result["voluntary_per_battle"]:14.2f} | {bad_text}', flush=True)
            print(f'Combined: {totals["wins"]}/{totals["n"]} = '
                  f'{totals["wins"] / totals["n"]:.1%} wins; {totals["ties"]} ties', flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
