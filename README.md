# Pokémon Reinforcement Learning — PPO prototypes

Experimental reinforcement-learning agents for **Pokémon Showdown**, using [`poke-env`](https://github.com/hsahovic/poke-env) and a **hand-implemented PPO** algorithm in PyTorch. The experiments and research question are still being developed.

## What is in this repository?

| Component | Purpose |
| --- | --- |
| `prototype.py` | Initial restricted 1v1 PPO experiment |
| `prototype_2v2.py` | Rich-observation fixed-team 2v2 PPO experiment |
| `pokemon_3v3_pool_ppo/` | Current 3v3 experiment: random teams of three from a fixed pool of 12 Pokémon; PPO and frozen historical-opponent self-play |
| `pokemon_1v1_pool_ppo/` | 1v1 with a random pool: each side gets one Pokémon drawn from a small fixed pool (default 6); same PPO/curriculum as 3v3, plus a `benchmark` timing command and per-matchup win-rate grid. |
| `audit_3v3_policy_fixed.py` | Compare a saved 3v3 PPO policy against an always-first-move baseline and inspect its decisions |
| `evaluate_*.py`, `inspect_*.py`, `play_vs_*.py`, `debug_2v2.py` | Earlier evaluations, diagnostic tools and human-play scripts (some target particular older prototypes) |
| `custom-formats.ts` | Earlier custom format definitions; the current combined 1v1/2v2/3v3 definitions are in `pokemon_3v3_pool_ppo/custom-formats-3v3.ts` |
| `models/` (if included) | Selected small inference checkpoints. Full training runs are kept out of Git. |

**Read `pokemon_3v3_pool_ppo/README.md` for full 3v3 CLI documentation**, and `README_2v2.md` for the earlier 2v2 experiment. Some older notes refer to the original author's paths or to tests not yet performed when those notes were written; use the setup below for new machines.

## Quick start (Windows PowerShell)

### Prerequisites

- Git and Python (developed with Python 3.12)
- Node.js and npm compatible with your checked-out Pokémon Showdown revision
- A local Pokémon Showdown server (do **not** expose the `--no-security` development server publicly)

From the cloned repository directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\pokemon_3v3_pool_ppo\requirements.txt
```

The `.venv` directory is intentionally not committed.

### Install and start Pokémon Showdown locally

```powershell
git clone https://github.com/smogon/pokemon-showdown.git pokemon-showdown
cd pokemon-showdown
npm install
cd ..

# Install custom formats used by our agents.
# If you already have custom formats, back up and MERGE rather than replacing them.
Copy-Item .\pokemon_3v3_pool_ppo\custom-formats-3v3.ts .\pokemon-showdown\config\custom-formats.ts -Force

cd pokemon-showdown
node pokemon-showdown start --no-security
```

Leave the Showdown server running. Open **another** PowerShell terminal in the repository root for the Python commands below. The local Showdown web client is normally at `http://localhost:8000`.

**Important:** For reproducible results, the whole group should record and use the same Pokémon Showdown commit. `poke-env` is pinned in the requirements file but Showdown changes over time.

### Verify the setup

```powershell
.\.venv\Scripts\python.exe .\pokemon_3v3_pool_ppo\offline_selftest.py
.\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\prototype_3v3.py check-pool
.\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\prototype_3v3.py smoke --battles 12
```

### Evaluate a pretrained 3v3 model

If `models/3v3_140200.pt` was included in the repository:

```powershell
.\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\prototype_3v3.py evaluate --weights .\models\3v3_140200.pt --opponent heuristic --battles 200 --seed 31415
.\.venv\Scripts\python.exe -u .\audit_3v3_policy_fixed.py --weights .\models\3v3_140200.pt --opponent typebot --battles 200 --compare-first-move
```

If no model is included, train one using the next section, or obtain a compatible checkpoint from a collaborator.

### Train a fresh 3v3 agent

```powershell
.\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\prototype_3v3.py train --steps 25000 --eval-every 5000 --eval-battles 100 --output runs\3v3_test --seed 42
```

To resume to 50,000 **total** decisions:

```powershell
.\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\prototype_3v3.py train --steps 50000 --eval-every 5000 --eval-battles 100 --output runs\3v3_test --resume runs\3v3_test\training_checkpoint.pt
```

The default curriculum mixes random, max-power and type-aware opponents. Starting at 15,000 decisions, it also samples frozen earlier PPO policies as historical opponents. The training run includes random exploration; evaluation selects the highest-probability legal action.

### Play against a saved agent

Open `http://localhost:8000` in your browser and choose a username. See `pokemon_3v3_pool_ppo/human_example_team.txt` for an importable team. With the server still running:

```powershell
.\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\play_vs_3v3.py --human YourShowdownUsername --weights .\models\3v3_140200.pt --team swampert,corviknight,garchomp
```

Change the username and weights path as needed. Use the `[Gen 9] RL 3v3 Singles` custom format.

## Results to date (exploratory; one training run)

The initial rich 2v2 PPO experiment achieved high win rates against its restricted scripted opponents and showed some strategic switching, but that does **not** imply general Pokémon competence.

In the 3v3 experiment, we trained one PPO run for 150,000 decisions with historical self-play. Results varied markedly between checkpoints. Among five evaluated checkpoints, the ~140,200-decision model scored **91/300 (30.3%)** against a held-out heuristic bot. The final 150,000-decision model scored **82/300 (27.3%)** against that bot. In the 150k action audit, it chose the first move slot **99.1%** of the time when legal and made **zero voluntary switches**. Its performance against a simple always-first-move baseline was very similar.

**Do not interpret these preliminary results as proof that self-play helps or hurts.** We have not yet run a matched no-self-play comparison, multiple training seeds, or a fully controlled evaluation across all variants. A key next step is auditing the observation/action representation and testing whether the fixed first-move ordering creates an exploitable shortcut.

## Collaborating and reproducibility

- Keep `main` for working code. Develop changes on separate branches and submit pull requests.
- Before merging, run the offline self-test and at least a short live smoke test if the change touches Showdown integration.
- Give each experiment a new output directory, record the random seed, configuration, Showdown commit, training budget, evaluation opponents and battle counts.
- **Don't commit** local environments, local Showdown checkout, entire training-run folders, secrets, or large transient checkpoints. Put small, selected inference weights in `models/` only when needed; include provenance and training settings.
- Preserve benchmark outputs in a small, curated CSV or Markdown summary when useful. Uncertainty, multiple seeds, and weak-baseline comparisons matter more than isolated high win rates.
- This code originated as AI-assisted experimental prototypes. All group members should audit, understand, test and adapt the work, and follow the module's academic-integrity and GenAI-disclosure rules.

## Dependencies and references

- [`poke-env`](https://github.com/hsahovic/poke-env)
- [Pokémon Showdown](https://github.com/smogon/pokemon-showdown)
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)

No existing third-party PPO training implementation is imported: `pokemon_3v3_pool_ppo/ppo.py` implements the learner with PyTorch tensor operations and automatic differentiation.
