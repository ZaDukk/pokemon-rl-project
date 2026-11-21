"""Audit whether 3v3 PPO has collapsed to move slot 0 (action 6).

Place this file in pokemon_1v1_rl_starter/ (beside pokemon_3v3_pool_ppo/),
then run with local Pokemon Showdown running:

 .\\.venv\\Scripts\\python.exe -u .\\audit_3v3_policy_fixed.py --weights runs\\3v3\\policy_00025000.pt --battles 200 --opponent typebot --compare-first-move

Read-only: does not modify your PPO model, save files or Showdown formats.
"""
from __future__ import annotations

import argparse
import random
import secrets
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "pokemon_3v3_pool_ppo"
sys.path.insert(0, str(PACKAGE if PACKAGE.is_dir() else HERE))

from features import bad_matchup, seed_all, species
from ppo import ActorCritic3, as_tensors
from prototype_3v3 import describe_action, make_env, reset_teams, step_safely


def evaluate_policy(label, weights, opponent, battles, seed, *, first_move_only=False, shared_env=None):
    seed_all(seed)
    rng = random.Random(seed)
    model = None
    if not first_move_only:
        model = ActorCritic3()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.eval()
    stats = Counter()
    slot_choices = Counter()
    species_moves = defaultdict(Counter)
    example_bad = []
    env = shared_env if shared_env is not None else make_env(opponent)
    try:
        for game in range(battles):
            obs, _, _, _ = reset_teams(env, rng=rng)
            done = False
            while not done:
                battle = env.env.battle1
                mask = np.asarray(obs["action_mask"], dtype=bool)
                can_move = bool(mask[6:10].any())
                can_switch = bool(mask[:6].any())
                with torch.no_grad():
                    if model is not None:
                        x, m = as_tensors(obs)
                        distribution, _ = model(x, m)
                        probs = distribution.probs.squeeze(0).cpu().numpy()
                        action = int(np.argmax(probs))
                    else:
                        # Honest, intentionally weak deterministic comparison:
                        # choose the first available attack, else the first legal action.
                        attacks = np.flatnonzero(mask[6:10])
                        action = int(attacks[0] + 6) if len(attacks) else int(np.flatnonzero(mask)[0])
                        probs = None
                if not mask[action]:
                    raise RuntimeError(f"Illegal selected action {action}; mask={mask.tolist()}")
                stats["decisions"] += 1
                if can_move:
                    stats["attack_available_decisions"] += 1
                    if mask[6]:
                        stats["first_move_legal"] += 1
                        if action == 6:
                            stats["first_move_chosen_when_legal"] += 1
                    if can_switch:
                        stats["free_choice_decisions"] += 1
                        if action < 6:
                            stats["voluntary_switches"] += 1
                        if probs is not None:
                            stats["switch_probability_sum"] += float(probs[:6].sum())
                            stats["first_move_probability_sum"] += float(probs[6])
                        if bad_matchup(battle):
                            stats["bad_matchup_opportunities"] += 1
                            if action < 6:
                                stats["bad_matchup_switches"] += 1
                            if probs is not None:
                                stats["bad_matchup_switch_probability_sum"] += float(probs[:6].sum())
                            if len(example_bad) < 10 and model is not None:
                                move_info=[]
                                moves=list(battle.active_pokemon.moves.values()) if battle.active_pokemon else []
                                for j in range(4):
                                    nm=moves[j].id if j<len(moves) else "(absent)"
                                    move_info.append(f"{nm}={probs[6+j]:.3f}")
                                example_bad.append((species(battle.active_pokemon),species(battle.opponent_active_pokemon),
                                                    describe_action(battle,action),float(probs[:6].sum()),move_info))
                if action < 6:
                    if can_move:
                        stats["voluntary_switch_all"] += 1
                    else:
                        stats["forced_switches"] += 1
                    slot_choices["switch"] += 1
                elif 6 <= action <= 9:
                    slot_choices[f"move{action - 6}"] += 1
                    if battle.active_pokemon:
                        species_moves[species(battle.active_pokemon)][describe_action(battle,action)] += 1
                else:
                    slot_choices[f"action{action}"] += 1
                obs, _, done, cap_forfeit = step_safely(env, action)
                stats["cap_forfeits"] += cap_forfeit
            stats["wins"] += int(bool(env.env.battle1.won))
            if (game + 1) % max(1, battles // 5) == 0:
                print(f"  {label}: completed {game + 1}/{battles} battles", flush=True)
    finally:
        if shared_env is None:
            env.close()

    def pct(num, den):
        return 100.0 * num / max(1, den)

    print(f"\n=== {label} vs {opponent} ({battles} battles) ===")
    print(f"Win rate: {stats['wins']}/{battles} ({pct(stats['wins'], battles):.1f}%)")
    print(f"Total decisions: {stats['decisions']} | cap forfeits: {stats['cap_forfeits']}")
    print(f"Chosen action slots: {dict(sorted(slot_choices.items()))}")
    print(f"First move selected when legal: {stats['first_move_chosen_when_legal']}/{stats['first_move_legal']} "
          f"({pct(stats['first_move_chosen_when_legal'],stats['first_move_legal']):.1f}%)")
    print(f"Voluntary switches: {stats['voluntary_switches']} in "
          f"{stats['free_choice_decisions']} free-choice decisions "
          f"({pct(stats['voluntary_switches'],stats['free_choice_decisions']):.1f}%)")
    print(f"Bad-matchup switches: {stats['bad_matchup_switches']} in "
          f"{stats['bad_matchup_opportunities']} bad-matchup opportunities")
    if model is not None:
        print(f"Average policy P(switch) when both legal: "
              f"{pct(stats['switch_probability_sum'],stats['free_choice_decisions']):.1f}%")
        print(f"Average policy P(move slot 0) when both legal: "
              f"{pct(stats['first_move_probability_sum'],stats['free_choice_decisions']):.1f}%")
        print(f"Average P(switch) in bad matchups: "
              f"{pct(stats['bad_matchup_switch_probability_sum'],stats['bad_matchup_opportunities']):.1f}%")
    print("Moves selected by species:")
    for mon, move_counts in sorted(species_moves.items()):
        print(f"  {mon:14s} {', '.join(f'{move} x{cnt}' for move,cnt in move_counts.most_common(5))}")
    if example_bad:
        print("Selected examples of bad matchups (probabilities are learned policy, not damage scores):")
        for own, opp, choice, switch_p, moves in example_bad[:6]:
            print(f"  {own} vs {opp}: chose {choice}; P(switch)={switch_p:.3f}; " + "; ".join(moves))
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', default='runs/3v3/policy_00025000.pt')
    p.add_argument('--opponent', choices=['random', 'maxpower', 'typebot', 'heuristic'], default='typebot')
    p.add_argument('--battles', type=int, default=100)
    p.add_argument('--seed', type=int, default=2028)
    p.add_argument('--compare-first-move', action='store_true', help='Also run an always-first-legal-attack baseline')
    a=p.parse_args()
    if a.battles<1:
        p.error('--battles must be >=1')
    # Create ONE environment for both passes. Re-creating an env after resetting
    # the global random seed can reuse Showdown guest names while the previous
    # websocket session still owns them (|nametaken|). Reuse avoids that race.
    # Seed guest-name generation independently of the experimental RNG; each
    # evaluate_policy() seeds battles and teams identically for both conditions.
    random.seed(secrets.randbits(64))
    env = make_env(a.opponent)
    try:
        evaluate_policy('PPO', a.weights, a.opponent, a.battles, a.seed, shared_env=env)
        if a.compare_first_move:
            evaluate_policy('Always-first-move baseline', a.weights, a.opponent,
                            a.battles, a.seed, first_move_only=True, shared_env=env)
    finally:
        env.close()


if __name__ == '__main__':
    main()
