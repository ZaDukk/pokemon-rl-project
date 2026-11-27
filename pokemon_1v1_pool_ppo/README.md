# 1v1 Pokémon RL: one Pokémon each from a fixed pool, PPO + historical self-play

A feasibility test that sits between the fixed-team prototypes and the 3v3 pool experiment. Each battle, **both players independently draw one Pokémon** uniformly from a small fixed pool (default **6**, mirror matches allowed). Everything else is the same as `../pokemon_3v3_pool_ppo`: the same hand-written PPO, scripted opponents, curriculum with frozen-checkpoint self-play, reward weights (hp 1 / faint 1 / win 5), 120-turn forfeit cap, and checkpoint/resume.

## What changed vs the 3v3 package

| | 3v3 pool | 1v1 pool (this) |
| --- | --- | --- |
| Team | 3 distinct from 12 | 1 from a pool of N (default 6) |
| Format | `[Gen 9] RL 3v3 Singles` | `[Gen 9] RL 1v1` (already in `custom-formats-3v3.ts`) |
| Observation | 2,823 features | 973 features (same encoder, `TEAM_SIZE = 1`) |
| PPO params | ~796k | ~322k (same 256-256 MLP) |
| Self-play starts at | 15,000 decisions | 7,500 (`--selfplay-start`) |
| Default eval | 60 battles / 5,000 decisions | 100 battles / 2,500 decisions |

- **Pool vs vocabulary.** The species/move/item vocabularies always cover all 12 sets, so the observation size stays the same whatever pool you choose. That makes it cheap to compare pool sizes. The *active pool* is chosen with `--pool` or `--pool-size`.
- **Default pool:** `garchomp, swampert, meowscarada, skeledirge, rotomwash, primarina`, which includes a Fire/Water/Grass triangle so type matchups matter. `--pool-size N` takes the first N of that ordering followed by the remaining six sets, so pool sizes 2 to 12 are nested.
- **No switching exists in 1v1.** The policy only ever chooses one of four moves, and the switch actions are always masked.
- **New `benchmark` command:** measures speed on *your* machine and projects training time.
- **New `evaluate --matrix N`:** plays every ordered matchup N times and prints a win-rate grid (agent's Pokémon × opponent's Pokémon).

## Run (from repository root, Showdown running with the custom formats installed)

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py .\pokemon_1v1_pool_ppo\offline_selftest.py
& $py .\pokemon_1v1_pool_ppo\prototype_1v1_pool.py info
& $py -u .\pokemon_1v1_pool_ppo\prototype_1v1_pool.py check-pool
& $py -u .\pokemon_1v1_pool_ppo\prototype_1v1_pool.py benchmark --battles 60 --ppo-steps 3000

# Train (steps are CUMULATIVE totals when resuming)
& $py -u .\pokemon_1v1_pool_ppo\prototype_1v1_pool.py train --steps 20000 --output runs\1v1_pool6
& $py -u .\pokemon_1v1_pool_ppo\prototype_1v1_pool.py train --steps 50000 --output runs\1v1_pool6 --resume runs\1v1_pool6\training_checkpoint.pt

# Held-out evaluation and the per-matchup grid
& $py -u .\pokemon_1v1_pool_ppo\prototype_1v1_pool.py evaluate --weights runs\1v1_pool6\policy.pt --opponent heuristic --battles 300
& $py -u .\pokemon_1v1_pool_ppo\prototype_1v1_pool.py evaluate --weights runs\1v1_pool6\policy.pt --opponent typebot --matrix 20 --json-out runs\1v1_pool6\matrix.json

# Other pools
& $py -u .\pokemon_1v1_pool_ppo\prototype_1v1_pool.py train --pool-size 5 --output runs\1v1_pool5
& $py -u .\pokemon_1v1_pool_ppo\prototype_1v1_pool.py train --pool garchomp,primarina,skeledirge,meowscarada,swampert --no-mirrors --output runs\1v1_custom
```

Pass the same `--pool`/`--pool-size`/`--no-mirrors` to `evaluate` that you trained with. Resume refuses to continue if the pool or hyperparameters differ. Plot with `plot_results.py --csv runs\1v1_pool6\results.csv --out runs\1v1_pool6\curves.png` (needs matplotlib). Play the bot with `play_vs_1v1_pool.py --human YourName --weights runs\1v1_pool6\policy.pt --mon garchomp`.

`results.csv` now also records `wall_seconds`, `train_seconds` and `eval_seconds`, so timing can go straight into the report.

## Measured timing (one machine; re-run `benchmark` on yours)

These numbers come from a 2-core Linux cloud VM with CPU-only PyTorch, Showdown commit `c7fa6f1`, `poke-env 0.16.1` and the default 6-Pokémon pool:

- Battles last about **3–5 turns**, which is about 3 agent decisions per battle.
- Simulator speed is about **30–40 battles/s** (about 150 decisions/s) with random actions.
- PPO training runs at about **130–140 decisions/s**, counting rollouts and PPO updates but not evaluation.
- Real runs took **1.8 min for 10k decisions** (≈3,600 battles) and **3.5 min for 20k** (≈7,000 battles), including evaluation every ~2,500 decisions.

| Decisions | ≈ Battles | Projected wall-clock (train + eval) |
| ---: | ---: | ---: |
| 10,000 | 3,600 | ~2 min |
| 50,000 | 18,000 | ~9 min |
| 100,000 | 36,000 | ~18 min |
| 250,000 | 90,000 | ~45 min |

Windows laptops are usually slower than this per core, so treat these figures as an order of magnitude. A multi-seed sweep (e.g. 3 seeds × 3 pool sizes × 100k decisions) is still a few hours, not days.

## Things to keep in mind for analysis

- **Matchup luck dominates.** With no switching, many pairings are decided by typing and speed before any choice is made. In a 20k-decision test run, the `--matrix 10` grid against typebot was 0% in some cells and 100% in others. For example, Skeledirge into Garchomp is close to unwinnable. A random-pool win rate therefore has a ceiling well below 100%, so compare against the **typebot-vs-typebot grid** or report per-matchup results rather than a single number.
- With only ~3 decisions per battle, each battle gives PPO far fewer learning signals than in 3v3. Compare runs by *battles* as well as by *decisions*.
- Mirror matches are about 1/N of games and are roughly symmetric. `--no-mirrors` removes them.
- The test run above was a functional check, not a tuned experiment; do not quote its win rates as results.
