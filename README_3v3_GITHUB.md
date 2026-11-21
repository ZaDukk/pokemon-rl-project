# 3v3 Pokémon RL — 12-Pokémon pool, PPO + historical self-play

**Status:** Live-validated experimental prototype. Offline self-test, all 12 Showdown team sets, smoke battles and a 150,000-decision training run completed successfully (November 2026). The trained policy remains strategically weak: it rarely varies moves and made no voluntary switches in the final evaluation. This code is a baseline for further research, not a strong general-purpose Pokémon bot.

## Why this setup?

- **3v3 Singles:** both players receive 3 distinct Pokémon drawn without replacement from the **same pool of 12**, in random order. First Pokémon leads. Team preview is off and Terastallization is disabled.
- **12-species pool:** Swampert, Corviknight, Meowscarada, Magnezone, Garchomp, Rotom-Wash, Skeledirge, Great Tusk, Dragapult, Primarina, Amoonguss, Kingambit. Each has a **fixed four-move set** in `pool.py`. This avoids having to solve team-building as well as battling.
- **Observable information:** 2,823 features including per-Pokémon types, stats, HP, known moves/PP, status, weather/terrain and action-aligned move/switch slots. **Unrevealed enemy team members and hidden stats are not used**. The representation uses fixed known vocabularies; unknowns receive their own bin.
- **Own PPO:** `ppo.py` implements clipped PPO, GAE, an actor-critic neural network, action masking, minibatches, Adam, and entropy regularization. PyTorch is used for tensor operations and differentiation; **no imported PPO trainer** or Stable-Baselines3.
- **Opponent curriculum:** random legal move bot (20%), max-base-power bot (30%), and a type-aware heuristic bot (50%). After 15,000 decisions, a **35% probability** of facing a frozen prior PPO policy is introduced, with the remaining proportions scaled down. Each opponent is chosen once **per battle**, not each turn.
- **Historical self-play** (not simultaneous self-play): only the learner updates. The opponent uses one of the most recent six **frozen checkpoint policies**, holding that policy fixed for a full battle. This is more stable for an initial experiment.
- **Independent evaluation:** `RandomPlayer`, `MaxBasePowerPlayer` and a **custom type-aware scripted bot**, 60 new randomized 3v3 games each checkpoint by default. `SimpleHeuristicsPlayer` is available for a fourth, **held-out evaluation** via the `evaluate` command, without training against it.
- **Checkpoints/resume:** includes optimizer state, training step count, Python/NumPy/Torch RNG states, and frozen opponent snapshots. Live Showdown battle state is **not serialised**; resuming starts a new battle. Checkpoint does not guarantee bit-for-bit identical continuation.

## 1. Add files to the repository

Keep this directory as `pokemon_3v3_pool_ppo/` within the repository. All relative paths in the commands below assume PowerShell is open at the **repository root**. Create and activate or directly use your own `.venv` as described in the [root README](../README.md). You do **not** need to replace `prototype_2v2.py`.

### Install the custom format (one time)

**In PowerShell, from starter project root:**

```powershell
if (Test-Path .\pokemon-showdown\config\custom-formats.ts) {
    Copy-Item .\pokemon-showdown\config\custom-formats.ts .\pokemon-showdown\config\custom-formats.backup.ts
}
Copy-Item .\pokemon_3v3_pool_ppo\custom-formats-3v3.ts .\pokemon-showdown\config\custom-formats.ts -Force
Select-String -Path .\pokemon-showdown\config\custom-formats.ts -Pattern 'RL 3v3 Singles'
```

**Restart** the local Showdown server. In its current server terminal use Ctrl+C, then:

```powershell
cd .\pokemon-showdown
node pokemon-showdown start --no-security
```

Open a **separate** PowerShell terminal at the starter project root to run Python. Do not run two Showdown instances on port 8000. The file includes the earlier 1v1 and 2v2 formats, but if you have additional custom formats, merge them instead of replacing your existing configuration.

### Check packages

Install the requirements into your local Python environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\pokemon_3v3_pool_ppo\requirements.txt
```

### Run checks (in order)

```powershell
# Offline: no Showdown needed
.\.venv\Scripts\python.exe .\pokemon_3v3_pool_ppo\offline_selftest.py
.\.venv\Scripts\python.exe .\pokemon_3v3_pool_ppo\prototype_3v3.py info

# Live: tests all 12 fixed sets with full battles
.\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\prototype_3v3.py check-pool

# Live: random legal actions and state/action handling
.\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\prototype_3v3.py smoke --battles 12
```

**If `check-pool` fails:** do not start training. Investigate the first Showdown exception (especially team illegality or unrecognized format). The 12 sets were validated in the original experiment, but new Showdown revisions can change legality.

## 2. First training experiment: 10,000 decisions

```powershell
.\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\prototype_3v3.py train --steps 10000 --eval-every 5000 --eval-battles 60 --output runs\3v3
```

At 10k decisions this is **scripted-opponent training**: historical self-play is configured to begin at 15k. This is intentional; it gives PPO a source of basic play before playing earlier versions of itself.

Expected output fields: `random`, `maxpower`, `typebot` win rates, number of completed training battles, decisions/sec, average switching and snapshots. **No win rate is promised**—this is a genuinely much harder task than the earlier fixed 2v2.

## 3. Continue to 60,000 decisions (enables historical self-play)

```powershell
.\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\prototype_3v3.py train --steps 60000 --eval-every 5000 --eval-battles 60 --output runs\3v3 --resume runs\3v3\training_checkpoint.pt
```

- From 15,000 decisions onward, about 35% of new training battles face a historical PPO snapshot (provided snapshots have been saved).
- The first `--steps` run starts from zero; the second `--steps` is a **cumulative total**, not 60k extra.
- The most recent model is in `runs\3v3\policy.pt`; preserved individual snapshots are `policy_XXXXXXXX.pt`. Selected inference checkpoints may also be shared separately in `models/`.
- `runs\3v3\results.csv` has evaluation metrics; `runs\3v3\run_config.json` records the run setup.
- A resumed run requires the **same training mode, rollout, learning rate and self-play schedule** as the original. To change these for a controlled experiment, create a separate run from scratch.

**Note on training time:** The model has about 796k parameters and the 3v3 task has larger teams. Don't assume the 2v2 throughput or success rate carries over. Evaluate intermediate results instead of committing to a long run immediately.

## 4. Independent evaluation against built-in `poke-env` agents

`SimpleHeuristicsPlayer` is deliberately **not** included in the default training mix. Evaluate it as a test of generalisation:

```powershell
.\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\prototype_3v3.py evaluate --weights runs\3v3\policy.pt --opponent heuristic --battles 240 --seed 2028
```

To inspect actual actions in a couple of battles:

```powershell
.\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\prototype_3v3.py evaluate --weights runs\3v3\policy.pt --opponent typebot --battles 24 --trace-battles 2
```

Available `--opponent`: `random`, `maxpower`, `typebot`, `heuristic`.

To evaluate an intermediate checkpoint, pass `--weights runs\3v3\policy_00020000.pt` (replace the step number with a file actually saved).

## 5. Optional: play against the learned model

Open `http://localhost:8000` in **Chrome** and prepare a three-Pokémon team in `[Gen 9] RL 3v3 Singles`. Import `human_example_team.txt` into the Showdown Teambuilder if you want a starting point. Choose a browser username, then run:

```powershell
.\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\play_vs_3v3.py --human HumanPlayer1 --weights runs\3v3\policy.pt --team swampert,corviknight,garchomp
```

Change `HumanPlayer1` to your browser username. Do not run until you have trained a model. The bot's team can be any 3 distinct names from the 12-species pool.

## 6. Optional plot

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib
.\.venv\Scripts\python.exe .\pokemon_3v3_pool_ppo\plot_results.py --csv runs\3v3\results.csv --out runs\3v3\learning_curves.png
```

## Experimental design for the coursework

**Main question to consider:** Does historical self-play improve generalisation and switching performance beyond scripted opponents in restricted 3v3 Pokémon battles?

Run a controlled *scripted-only* baseline, changing only the opponent curriculum, in a **separate results directory**:

```powershell
.\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\prototype_3v3.py train --steps 60000 --opponent typebot --eval-every 5000 --output runs\3v3_typebot_only
```

Also compare `--opponent curriculum` from the start. For a rigorous report, repeat each setting under **at least three training seeds** and run evaluation against the same fixed unseen opponent choices (e.g. `heuristic` with seed 2028). Report means, variation, and results across increasing decisions—not just the best checkpoint. Note that Python RNG seeding does **not** fully control the Showdown engine's game randomness.

A better test later is to hold out some species or team compositions during training; this package **does not implement a holdout roster split** yet. Performance on a new random combination of the same 12 Pokémon is *team-composition generalisation*, not generalisation to unseen species.

## Limitations and caution

- These scripts have been tested against the original local Showdown server. Run `check-pool` on new checkouts: Pokémon Showdown updates may change format validity or behaviour.
- Hidden-information policy: the opponent sees only revealed Pokémon and moves; the learner receives full own-team data and public data about the opponent. Fixed pool knowledge can be used as prior information but the encoder does not expose an opponent's unrevealed selected Pokémon.
- The script intentionally does **not** use pretrained RL weights. Previous 2v2 `policy.pt` checkpoints are incompatible.
- `--opponent historical` alone starts against the custom typebot until some snapshots are available; use `curriculum` for the intended staged experiment.
- There is a **120-turn safety cap**: when reached, the learner **forfeits using a real Showdown action**, counting it as a loss. It is *not* a silent episode reset. Check the `typebot_cap_forfeits` metric; frequent cap forfeits signal a flawed policy and should be investigated.
- The type-aware bot approximates potential damage with power, STAB and type multipliers, not Showdown's full damage formula. It is an informative baseline, not an optimal opponent.
- The snapshot history is capped at six policies in memory. Snapshot evaluation and selection is part of the training process; these are not independent tests.
- More input features and PPO parameters do **not** guarantee improved performance. The earlier rich 2v2 performance was on a much narrower task; do not compare 2v2 and 3v3 win rates as though they were equivalent.
- All group members should be able to explain and audit the algorithm and experiments. **Coursework specification requires disclosure of GenAI assistance**, reproducible source code, meaningful comparisons and a report of no more than seven pages. Do not claim any results until the corresponding tests actually run.


## Experimental findings as of 21 November 2026

One exploratory 3v3 training run was continued to **150,000 decisions** (with historical self-play beginning after 15,000 decisions). Training win rates varied substantially across checkpoints. In a 300-battle held-out heuristic evaluation, checkpoint 140,200 achieved **91/300 wins (30.3%)** while checkpoint 150,000 achieved **82/300 (27.3%)**. These figures come from one run and do not establish statistical superiority between checkpoints.

The 150,000-decision policy showed **99.1% first-move preference when the first move was legal** and **zero voluntary switches** in a 300-battle action audit against the type-aware bot. Its win rate was **73/300 (24.3%)**, while an always-first-move comparison scored **71/300 (23.7%)** in that audit. A second audit with different game randomness showed similarly close rates. **We have not established that historical self-play improved performance**; a controlled no-self-play comparison is still needed.

Keep model snapshots and run logs on disk outside Git unless choosing a small set to share deliberately. Use the root README for GitHub collaboration and setup from a fresh clone.

## Source files

- `pool.py` — fixed Pokémon sets and uniform 3-of-12 sampling
- `features.py` — fixed-size visible-state encoder; species/move vocabularies
- `ppo.py` — **our PPO implementation** (no SB3)
- `opponents.py` — fixed bots, type bot and historical checkpoint opponent
- `prototype_3v3.py` — live environment, training/evaluation, checkpoints/resume
- `offline_selftest.py` — fake-state observation safety and PPO gradient checks
- `custom-formats-3v3.ts` — replaces/merges Showdown's custom-format config
- `play_vs_3v3.py` — browser challenge against saved model
- `human_example_team.txt` — importable Showdown human team
- `plot_results.py` — optional learning-curve plot

### References

- [`poke-env` documentation](https://poke-env.readthedocs.io/en/stable/) (environment, baseline agents and action masks).
- [`poke-env` self-play example](https://poke-env.readthedocs.io/en/latest/examples/self_play.html) (different training architecture: simultaneous shared-policy self-play, requires SB3; **not imported or used here**).
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms* (PPO).
- Schulman et al. (2015), *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (GAE).
