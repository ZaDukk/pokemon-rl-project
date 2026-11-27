"""Gen 9 1v1 Singles, each side's Pokemon drawn from a small fixed pool + HAND-WRITTEN PPO.

Same PPO, opponents, curriculum, checkpointing and reward weights as the 3v3
package (../pokemon_3v3_pool_ppo), but with ONE Pokemon per side. Each battle,
both players independently draw one Pokemon uniformly from the pool (default 6,
mirror matches allowed). Uses the existing '[Gen 9] RL 1v1' custom format.

  python prototype_1v1_pool.py info
  python prototype_1v1_pool.py check-pool
  python prototype_1v1_pool.py benchmark --battles 60 --ppo-steps 2048
  python prototype_1v1_pool.py train --steps 20000 --eval-every 2500 --output runs/1v1_pool6
  python prototype_1v1_pool.py train --steps 50000 --output runs/1v1_pool6 --resume runs/1v1_pool6/training_checkpoint.pt
  python prototype_1v1_pool.py evaluate --weights runs/1v1_pool6/policy.pt --opponent heuristic --battles 300
  python prototype_1v1_pool.py evaluate --weights runs/1v1_pool6/policy.pt --opponent typebot --matrix 10

Every command accepts --pool (comma-separated names) or --pool-size N.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv, SingleAgentWrapper
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer

from pool import (SPECIES, all_matchups, pack_team, parse_pool, sample_matchup,
                  sample_mon, static_checks)
from features import N_FEATURES, TEAM_SIZE, battle_features, seed_all, species
from opponents import MixedOpponent, TypeAwarePlayer
from ppo import ACTION_COUNT, ActorCritic, choose_action, as_tensors, ppo_update

FORMAT = 'gen9rl1v1'
MAX_TURNS = 120
EVAL_OPPONENTS = ['random', 'maxpower', 'typebot']


class PoolOneVsOneEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(low=-1.0, high=1.0, shape=(N_FEATURES,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return battle_features(battle)

    def calc_reward(self, battle: AbstractBattle) -> float:
        # Same reward WEIGHTS as the 2v2/3v3 experiments, team size 1.
        return self.reward_computing_helper(
            battle, number_of_pokemons=TEAM_SIZE,
            hp_value=1.0, fainted_value=1.0, victory_value=5.0)


def make_env(opponent: str, pool) -> SingleAgentWrapper:
    base = PoolOneVsOneEnv(
        battle_format=FORMAT, team=pack_team(pool[0]),
        choose_on_teampreview=False, log_level=40,
        open_timeout=10.0, challenge_timeout=45.0,
    )
    base.agent2.update_team(pack_team(pool[1]))
    opponents = {
        'random': RandomPlayer, 'maxpower': MaxBasePowerPlayer,
        'heuristic': SimpleHeuristicsPlayer, 'typebot': TypeAwarePlayer,
        'curriculum': MixedOpponent,
    }
    cls = opponents.get(opponent)
    if cls is None:
        raise ValueError(f'Unknown opponent {opponent}')
    return SingleAgentWrapper(base, cls(battle_format=FORMAT, start_listening=False))


def reset_mons(env, pool, rng=None, *, ours=None, theirs=None, mirrors=True):
    if ours is None or theirs is None:
        o, t = sample_matchup(pool, rng, mirrors=mirrors)
        ours, theirs = ours or o, theirs or t
    ours, theirs = sample_mon(pool, name=ours), sample_mon(pool, name=theirs)
    env.env.agent1.update_team(pack_team(ours))
    env.env.agent2.update_team(pack_team(theirs))
    obs, info = env.reset()
    return obs, info, ours, theirs


def step_safely(env, action):
    """At MAX_TURNS, end the battle with a REAL Showdown forfeit (counted as a loss).

    Never reset an unfinished env, which can hang the poke-env event loop.
    """
    obs, reward, terminated, truncated, info = env.step(np.int64(action))
    done = bool(terminated or truncated)
    forced = 0
    if not done and env.env.battle1.turn >= MAX_TURNS:
        obs2, r2, t2, tc2, _ = env.step(np.int64(-1))
        if not (t2 or tc2):
            raise RuntimeError('Forfeit at the turn cap failed: battle still live. Do not reset.')
        obs, reward, done, forced = obs2, float(reward) + float(r2), True, 1
    return obs, float(reward), done, forced


def describe_action(battle, a):
    if 6 <= a < 10 and battle.active_pokemon:
        moves = list(battle.active_pokemon.moves.values())
        return moves[a - 6].id if a - 6 < len(moves) else 'MOVE?'
    return f'action {a}'


def play_one(env, model, pool, rng, *, ours=None, theirs=None, mirrors=True, trace=False):
    obs, _, ours, theirs = reset_mons(env, pool, rng, ours=ours, theirs=theirs, mirrors=mirrors)
    decisions = forced = 0
    while True:
        b = env.env.battle1
        action, _, _ = choose_action(model, obs, deterministic=True)
        if not obs['action_mask'][action]:
            raise RuntimeError(f'Masked policy selected illegal action {action}')
        if trace:
            print(f'  [turn {b.turn}] {species(b.active_pokemon)} vs '
                  f'{species(b.opponent_active_pokemon)}: {describe_action(b, action)}', flush=True)
        obs, _, done, f = step_safely(env, action)
        decisions += 1
        forced += f
        if done:
            break
    b = env.env.battle1
    return dict(ours=ours, theirs=theirs, win=int(bool(b.won)),
                tie=int(not b.won and not b.lost), turns=b.turn,
                decisions=decisions, cap_forfeit=forced)


def summarise(games, elapsed):
    n = max(1, len(games))
    by = defaultdict(lambda: [0, 0])
    for g in games:
        by[f"{g['ours']} vs {g['theirs']}"][0] += g['win']
        by[f"{g['ours']} vs {g['theirs']}"][1] += 1
    return {
        'win_rate': sum(g['win'] for g in games) / n,
        'ties': sum(g['tie'] for g in games),
        'battles': len(games),
        'turns_per_battle': sum(g['turns'] for g in games) / n,
        'decisions_per_battle': sum(g['decisions'] for g in games) / n,
        'cap_forfeits': sum(g['cap_forfeit'] for g in games),
        'battles_per_second': len(games) / max(elapsed, 1e-9),
        'decisions_per_second': sum(g['decisions'] for g in games) / max(elapsed, 1e-9),
        'elapsed_seconds': round(elapsed, 2),
        'matchups': {k: {'wins': v[0], 'games': v[1]} for k, v in sorted(by.items())},
    }


def evaluate(env, model, pool, games, *, seed=2027, mirrors=True, verbose=False, trace=0):
    """Random matchups from the pool with a fixed seed; no weight updates."""
    rng = random.Random(seed)
    start = time.perf_counter()
    results = []
    for i in range(games):
        results.append(play_one(env, model, pool, rng, mirrors=mirrors, trace=i < trace))
        if verbose and (i + 1) % max(1, games // 10) == 0:
            print(f'  evaluated {i + 1}/{games}', flush=True)
    return summarise(results, time.perf_counter() - start)


def evaluate_matrix(env, model, pool, per_matchup, *, mirrors=True, seed=2027):
    """Play EVERY ordered matchup `per_matchup` times (balanced; good for analysis)."""
    rng = random.Random(seed)
    start = time.perf_counter()
    results = []
    pairs = all_matchups(pool, mirrors=mirrors)
    for k, (a, b) in enumerate(pairs):
        for _ in range(per_matchup):
            results.append(play_one(env, model, pool, rng, ours=a, theirs=b))
        print(f'  matchup {k + 1}/{len(pairs)} {a} vs {b}: '
              f'{sum(r["win"] for r in results[-per_matchup:])}/{per_matchup}', flush=True)
    return summarise(results, time.perf_counter() - start)


def print_matrix(summary, pool):
    m = summary['matchups']
    w = max(len(p) for p in pool) + 1
    print('\nWin rate of agent (rows) vs opponent Pokemon (columns):')
    print(' ' * w + ''.join(f'{p[:9]:>10}' for p in pool))
    for a in pool:
        cells = []
        for b in pool:
            r = m.get(f'{a} vs {b}')
            cells.append(f'{r["wins"] / r["games"]:>10.0%}' if r else f'{"-":>10}')
        print(f'{a:<{w}}' + ''.join(cells))


# ---------------------------------------------------------------- training
def empty_batch():
    return {x: [] for x in ('obs', 'mask', 'action', 'logp', 'value', 'reward', 'done')}


def save_checkpoint(out, model, optimizer, steps, battles, opponent, *, pool, resume_config):
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f'policy_{steps:08d}.pt')
    torch.save(model.state_dict(), out / 'policy.pt')
    torch.save({
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'steps': steps, 'battles': battles,
        'snapshots': [(s, m.state_dict()) for s, m in opponent.snapshots],
        'py_random': random.getstate(), 'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
        'n_features': N_FEATURES, 'vocab': SPECIES, 'pool': tuple(pool), 'format': FORMAT,
        'resume_config': resume_config,
    }, out / 'training_checkpoint.pt')


def train(args, pool):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = battles = 0
    resume_config = dict(opponent=args.opponent, selfplay_start=args.selfplay_start,
                         historical_prob=args.historical_prob, rollout=args.rollout,
                         lr=args.lr, pool=list(pool), mirrors=not args.no_mirrors)
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if ckpt['n_features'] != N_FEATURES or tuple(ckpt['vocab']) != SPECIES:
            raise ValueError('Cannot resume: observation encoding changed.')
        if ckpt.get('resume_config', {}) != resume_config:
            raise ValueError(f'Resume config differs! Saved {ckpt.get("resume_config")}; '
                             f'now {resume_config}. Keep pool/hyperparameters the same.')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        steps, battles = int(ckpt['steps']), int(ckpt['battles'])
        print(f'Resuming at {steps} decisions (a fresh battle starts).', flush=True)
    meta = dict(poke_env='0.16.1', format=FORMAT, pool=list(pool), team_size=1,
                mirrors=not args.no_mirrors, state_features=N_FEATURES,
                action_count=ACTION_COUNT, reward=dict(hp=1, faint=1, win=5),
                algorithm='our PPO with clipped loss, GAE, legal action masking',
                args=vars(args))
    (out / 'run_config.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    train_env = make_env('curriculum', pool)
    eval_envs = {name: make_env(name, pool) for name in EVAL_OPPONENTS}
    mixer = train_env.opponent
    assert isinstance(mixer, MixedOpponent)
    if ckpt is not None:
        for old_steps, state in ckpt.get('snapshots', []):
            m = ActorCritic(); m.load_state_dict(state); m.eval()
            mixer.snapshots.append((old_steps, m))
        random.setstate(ckpt['py_random'])
        np.random.set_state(ckpt['numpy_random'])
        torch.set_rng_state(ckpt['torch_random'])

    results = out / 'results.csv'
    fields = ['steps', 'train_battles', 'wall_seconds', 'train_seconds', 'eval_seconds',
              'train_decisions_per_second', 'vs_random', 'vs_maxpower', 'vs_typebot',
              'typebot_turns_per_battle', 'typebot_cap_forfeits', 'snapshots',
              'train_random', 'train_maxpower', 'train_typebot', 'train_historical',
              'ppo_policy_loss', 'ppo_value_loss', 'ppo_entropy', 'ppo_approx_kl']
    if not results.exists() or steps == 0:
        with results.open('w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
    train_secs = eval_secs = 0.0
    started = time.perf_counter()
    next_eval = ((steps // args.eval_every) + 1) * args.eval_every
    mirrors = not args.no_mirrors
    start_steps = steps  # so decisions/s after --resume only counts this session

    def reset_train():
        mixer.choose_for_next_battle(steps, strategy=args.opponent, start=args.selfplay_start,
                                     historical_prob=args.historical_prob)
        return reset_mons(train_env, pool, mirrors=mirrors)[0]

    try:
        print(f'1v1 pool: {list(pool)} ({len(pool)} Pokemon, '
              f'{len(all_matchups(pool, mirrors=mirrors))} ordered matchups); '
              f'{N_FEATURES} features; {sum(p.numel() for p in model.parameters()):,} '
              f'PPO parameters.', flush=True)
        if steps == 0:
            print('Untrained baselines (random legal action agent):', flush=True)
            for label, e in eval_envs.items():
                r = evaluate(e, None, pool, args.eval_battles, seed=args.eval_seed, mirrors=mirrors)
                print(f'  vs {label}: {r["win_rate"]:.1%} wins', flush=True)
        obs = reset_train()
        while steps < args.steps:
            batch = empty_batch()
            t0 = time.perf_counter()
            for _ in range(min(args.rollout, args.steps - steps)):
                action, logp, value = choose_action(model, obs)
                batch['obs'].append(obs['observation'].copy())
                batch['mask'].append(obs['action_mask'].copy())
                batch['action'].append(action)
                batch['logp'].append(logp)
                batch['value'].append(value)
                obs, reward, done, _ = step_safely(train_env, action)
                batch['reward'].append(float(reward))
                batch['done'].append(bool(done))
                steps += 1
                if done:
                    battles += 1
                    obs = reset_train()
            if batch['done'][-1]:
                bootstrap = 0.0
            else:
                with torch.no_grad():
                    x, mask = as_tensors(obs)
                    bootstrap = float(model(x, mask)[1].item())
            losses = ppo_update(model, optimizer, batch, bootstrap, gamma=args.gamma,
                                gae_lambda=args.gae_lambda, clip=args.clip, epochs=args.epochs,
                                minibatch=args.minibatch, entropy_coef=args.entropy)
            train_secs += time.perf_counter() - t0
            if steps >= next_eval or steps >= args.steps:
                t1 = time.perf_counter()
                metrics = {n: evaluate(e, model, pool, args.eval_battles, seed=args.eval_seed,
                                       mirrors=mirrors) for n, e in eval_envs.items()}
                eval_secs += time.perf_counter() - t1
                wall = time.perf_counter() - started
                rate = (steps - start_steps) / max(train_secs, 1e-9)
                scores = ' | '.join(f'{n} {metrics[n]["win_rate"]:.1%}' for n in EVAL_OPPONENTS)
                print(f'{steps:>8} decisions / {battles:>5} battles | {scores} | '
                      f'{rate:.1f} train steps/s | wall {wall / 60:.1f} min '
                      f'(eval {eval_secs / 60:.1f}) | snapshots {len(mixer.snapshots)}', flush=True)
                row = dict(steps=steps, train_battles=battles, wall_seconds=round(wall, 1),
                           train_seconds=round(train_secs, 1), eval_seconds=round(eval_secs, 1),
                           train_decisions_per_second=round(rate, 2),
                           vs_random=metrics['random']['win_rate'],
                           vs_maxpower=metrics['maxpower']['win_rate'],
                           vs_typebot=metrics['typebot']['win_rate'],
                           typebot_turns_per_battle=metrics['typebot']['turns_per_battle'],
                           typebot_cap_forfeits=metrics['typebot']['cap_forfeits'],
                           snapshots=len(mixer.snapshots),
                           **{f'train_{k}': v for k, v in mixer.mode_counts.items()},
                           **{f'ppo_{k}': losses.get(k, 0.0) for k in
                              ('policy_loss', 'value_loss', 'entropy', 'approx_kl')})
                with results.open('a', newline='', encoding='utf-8') as f:
                    csv.DictWriter(f, fieldnames=fields).writerow(row)
                mixer.add_snapshot(model, steps)
                save_checkpoint(out, model, optimizer, steps, battles, mixer,
                                pool=pool, resume_config=resume_config)
                next_eval = ((steps // args.eval_every) + 1) * args.eval_every
        print(f'Finished. Checkpoints: {out / "policy_*.pt"}; logs: {results}; '
              f'elapsed {(time.perf_counter() - started) / 60:.1f} min.', flush=True)
    finally:
        train_env.close()
        for e in eval_envs.values():
            e.close()


def benchmark(args, pool):
    """Answer 'how long will this take?' on THIS machine.

    1) simulator throughput with random actions vs each scripted bot;
    2) real PPO training throughput (rollouts + updates) for --ppo-steps decisions;
    3) projected wall-clock time for common run lengths, including evaluation.
    """
    mirrors = not args.no_mirrors
    report = {}
    for name in EVAL_OPPONENTS:
        env = make_env(name, pool)
        try:
            r = evaluate(env, None, pool, args.battles, seed=args.seed, mirrors=mirrors)
        finally:
            env.close()
        report[name] = r
        print(f'random agent vs {name:<8}: {r["battles_per_second"]:.2f} battles/s, '
              f'{r["decisions_per_second"]:.1f} decisions/s, {r["turns_per_battle"]:.1f} turns/battle, '
              f'agent win {r["win_rate"]:.0%}', flush=True)
    # Short real training loop: measures PPO overhead + curriculum opponents.
    env = make_env('curriculum', pool)
    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    mixer = env.opponent
    steps = battles = 0
    try:
        mixer.choose_for_next_battle(0, strategy='curriculum', start=10**9, historical_prob=0)
        obs = reset_mons(env, pool, mirrors=mirrors)[0]
        t0 = time.perf_counter()
        while steps < args.ppo_steps:
            batch = empty_batch()
            for _ in range(min(256, args.ppo_steps - steps)):
                a, lp, v = choose_action(model, obs)
                batch['obs'].append(obs['observation'].copy()); batch['mask'].append(obs['action_mask'].copy())
                batch['action'].append(a); batch['logp'].append(lp); batch['value'].append(v)
                obs, rew, done, _ = step_safely(env, a)
                batch['reward'].append(rew); batch['done'].append(done)
                steps += 1
                if done:
                    battles += 1
                    mixer.choose_for_next_battle(0, strategy='curriculum', start=10**9, historical_prob=0)
                    obs = reset_mons(env, pool, mirrors=mirrors)[0]
            ppo_update(model, opt, batch, 0.0)
        train_t = time.perf_counter() - t0
    finally:
        env.close()
    rate = steps / train_t
    eval_bps = float(np.mean([r['battles_per_second'] for r in report.values()]))
    print(f'\nPPO training: {steps} decisions / {battles} battles in {train_t:.1f}s = '
          f'{rate:.1f} decisions/s ({steps / max(battles, 1):.1f} decisions per battle)')
    print(f'\nProjected wall-clock (eval {args.eval_battles} battles x 3 bots every '
          f'{args.eval_every} decisions):')
    for total in (10_000, 25_000, 50_000, 100_000, 250_000):
        n_evals = -(-total // args.eval_every)
        ev = n_evals * 3 * args.eval_battles / eval_bps
        print(f'  {total:>7,} decisions: train {total / rate / 60:6.1f} min + eval '
              f'{ev / 60:5.1f} min = {(total / rate + ev) / 60:6.1f} min')
    print('Historical self-play opponents add a small extra cost per opponent move.')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--pool', default=None,
                        help='Comma-separated Pokemon, e.g. garchomp,swampert,primarina '
                             '(default: 6-mon DEFAULT_POOL in pool.py)')
    common.add_argument('--pool-size', type=int, default=None,
                        help='Use the first N of the default ordering (2..12)')
    common.add_argument('--no-mirrors', action='store_true',
                        help='Never give both players the same species')
    sub.add_parser('info', parents=[common], help='Offline: pool and feature sizes')
    c = sub.add_parser('check-pool', parents=[common], help='Every pool Pokemon in a live battle')
    c.add_argument('--seed', type=int, default=42)
    b = sub.add_parser('benchmark', parents=[common], help='Measure speed and project training time')
    b.add_argument('--battles', type=int, default=40, help='random-agent battles per bot')
    b.add_argument('--ppo-steps', type=int, default=2048)
    b.add_argument('--eval-every', type=int, default=2500)
    b.add_argument('--eval-battles', type=int, default=100)
    b.add_argument('--seed', type=int, default=42)
    t = sub.add_parser('train', parents=[common], help='Train PPO (scripted curriculum + historical self-play)')
    t.add_argument('--steps', type=int, default=20000)
    t.add_argument('--rollout', type=int, default=256)
    t.add_argument('--eval-every', type=int, default=2500)
    t.add_argument('--eval-battles', type=int, default=100)
    t.add_argument('--eval-seed', type=int, default=2027)
    t.add_argument('--output', default='runs/1v1_pool')
    t.add_argument('--resume', default=None)
    t.add_argument('--opponent', choices=['curriculum', 'random', 'maxpower', 'typebot', 'historical'],
                   default='curriculum')
    t.add_argument('--selfplay-start', type=int, default=7500,
                   help='Decisions before frozen-PPO opponents are mixed in (3v3 uses 15000)')
    t.add_argument('--historical-prob', type=float, default=.35)
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--gamma', type=float, default=.99)
    t.add_argument('--gae-lambda', type=float, default=.95)
    t.add_argument('--clip', type=float, default=.2)
    t.add_argument('--epochs', type=int, default=4)
    t.add_argument('--minibatch', type=int, default=64)
    t.add_argument('--entropy', type=float, default=.01)
    t.add_argument('--seed', type=int, default=42)
    e = sub.add_parser('evaluate', parents=[common], help='Evaluate saved weights (no training)')
    e.add_argument('--weights', required=True)
    e.add_argument('--opponent', choices=['random', 'maxpower', 'typebot', 'heuristic'], default='typebot')
    e.add_argument('--battles', type=int, default=200)
    e.add_argument('--matrix', type=int, default=0, metavar='N',
                   help='Instead of random matchups, play every ordered matchup N times and print a grid')
    e.add_argument('--seed', type=int, default=2027)
    e.add_argument('--trace-battles', type=int, default=0)
    e.add_argument('--json-out', default=None)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        pool = parse_pool(args.pool, args.pool_size)
    except ValueError as err:
        parser.error(str(err))
    mirrors = not args.no_mirrors
    if args.command == 'info':
        print(json.dumps(dict(**static_checks(), active_pool=list(pool),
                              ordered_matchups=len(all_matchups(pool, mirrors=mirrors)),
                              N_FEATURES=N_FEATURES, N_ACTIONS=ACTION_COUNT,
                              ppo_parameters=sum(p.numel() for p in ActorCritic().parameters())),
                         indent=2))
        return
    seed_all(getattr(args, 'seed', 42))
    if args.command == 'train':
        if min(args.steps, args.rollout, args.eval_every, args.eval_battles) <= 0:
            parser.error('steps, rollout, eval-every and eval-battles must be positive')
        if not 0 <= args.historical_prob <= 1:
            parser.error('--historical-prob must be 0 to 1')
        train(args, pool)
    elif args.command == 'benchmark':
        benchmark(args, pool)
    elif args.command == 'check-pool':
        env = make_env('random', pool)
        try:
            for i, mon in enumerate(pool):
                g = play_one(env, None, pool, random.Random(i), ours=mon, theirs=pool[(i + 1) % len(pool)])
                print(f'  ok: {g["ours"]} vs {g["theirs"]} ({g["turns"]} turns)', flush=True)
            print(f'All {len(pool)} pool Pokemon passed Showdown validation and finished a battle.')
        finally:
            env.close()
    elif args.command == 'evaluate':
        model = ActorCritic()
        model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True))
        model.eval()
        env = make_env(args.opponent, pool)
        try:
            if args.matrix > 0:
                r = evaluate_matrix(env, model, pool, args.matrix, mirrors=mirrors, seed=args.seed)
            else:
                r = evaluate(env, model, pool, args.battles, seed=args.seed, mirrors=mirrors,
                             verbose=True, trace=args.trace_battles)
        finally:
            env.close()
        print(json.dumps({k: v for k, v in r.items() if k != 'matchups'}, indent=2))
        print_matrix(r, pool)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(r, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
