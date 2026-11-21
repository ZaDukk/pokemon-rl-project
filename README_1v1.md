# Pokémon Showdown 1v1 PPO feasibility test

**What this is:** A small prototype for CM32032. Each player has exactly one
fixed Conkeldurr with Drain Punch, Mach Punch, Bulk Up, and Protect. A
self-implemented masked PPO policy trains against a random-moving opponent.
This is **not** sufficient as a final coursework environment or experiment.

## Prerequisites

- Python 3.11 or 3.12, Git, Node.js **22.18.0 or newer**.
- `poke-env==0.16.1` (as pinned in `requirements.txt`).
- A local Pokémon Showdown server. **Do not expose a `--no-security` server to the internet.**

### 1. Install Pokémon Showdown

In a terminal:

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
```

Copy `pokemon_1v1_rl_starter/custom-formats.ts` into
`pokemon-showdown/config/custom-formats.ts` before starting the server.
This defines `[Gen 9] RL 1v1`, an ordinary singles battle with exactly one
Pokémon on each side, no Terastallization, and **no Team Preview**.

Copy the server config example, if it does not already exist:

- PowerShell: `Copy-Item config/config-example.js config/config.js`
- macOS/Linux: `cp config/config-example.js config/config.js`

Start Showdown and keep this terminal open:

```bash
node pokemon-showdown start --no-security
```

Default server endpoint is `ws://localhost:8000/showdown/websocket`.

### 2. Install Python dependencies

Open **another terminal** in `pokemon_1v1_rl_starter`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`.
If PowerShell blocks Activate.ps1, run `./.venv/Scripts/python.exe` directly.

### 3. Verify Showdown and benchmark simulation

```bash
python prototype.py smoke --battles 40
```

The output should report random-policy win rate, decisions per second,
battles per second, average decisions per battle, and wall-clock time.
Random vs random mirror should typically be around 50% over many battles,
but 40 battles has considerable sampling noise.

### 4. Test a very short training run

```bash
python prototype.py train --steps 1000 --eval-every 500 --eval-battles 30
```

Then a longer training attempt:

```bash
python prototype.py train --steps 10000 --eval-every 2000 --eval-battles 100
```

Training prints decisions/s (excluding evaluation overhead) and evaluation
win rates. It writes `results/results.csv` and `results/policy.pt`.

Training decisions are **individual move choices**, not complete battles.
The evaluation uses a greedy policy and win rate against the same random
opponent distribution; this is a feasibility check, not a rigorous score.

## Project design details

- Action space: the library's 26 Gen-9 singles action indices; legal action
  masking restricts decisions to the available moves (normally indices 6-9).
- Observation space: both Pokémon's HP fractions, visible attack/defence
  boosts for both players, and battle turn (seven numerical features).
- Reward: change in own minus opponent HP, fainted bonuses, and +5/-5
  terminal win/loss reward (implemented using `reward_computing_helper`).
- Algorithm: independent minimal PyTorch PPO with GAE, clipped policy
  objective, entropy regularisation, value regression, and action masking.
- Teams: both sides of the `SinglesEnv` use the same fixed Conkeldurr team.
  This is intentional; a later prototype can support distinct teams.

## How to interpret the benchmark

1. If `smoke` fails to start, first check the custom format, local server,
   Node.js version, team validation, and Python dependencies.
2. If random vs random doesn't finish, look for Team Preview errors or invalid
   action exceptions (the format intentionally omits Team Preview).
3. If training is slow, compare *decisions/s* and *battles/s*; a fast neural
   network cannot overcome a slow battle server.
4. If wins don't increase, that alone does not prove the environment is too
   hard. Verify correct rewards, action masks, PPO losses, and baseline
   strength. A single seed is not statistically persuasive.
5. Repeat longer runs with several seeds before comparing learning methods.

## Limitations and reproducibility

The example is based on `poke-env` 0.16.1 documentation and published source.
It has been checked for Python syntax, but **not live-tested against a
running Showdown server** in the file-generation environment. Minor version
or simulator changes may require adjustments.

Showdown battle RNG is not fully fixed by `--seed`; this benchmark prioritises
useful throughput measurement over perfectly reproducible game outcomes.

Before a full coursework submission, record and pin the precise Showdown Git
commit, add multiple random seeds and stronger baselines, and test the PPO
implementation on a separate environment. The coursework requires your own
function-approximated RL implementation, which this prototype starts.

Relevant documentation:
- https://poke-env.readthedocs.io/en/latest/examples/reinforcement_learning.html
- https://poke-env.readthedocs.io/en/latest/modules/env.html
- https://github.com/smogon/pokemon-showdown/blob/master/config/CUSTOM-RULES.md
