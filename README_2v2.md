# Pokémon 2v2 singles switching feasibility experiment

This is a follow-on to your **working** `pokemon_1v1_rl_starter` folder. Keep the
original `prototype.py`, `.venv`, `results/policy.pt`, and Showdown installation.

## Design

Two Pokémon are brought by each player, but **only one Pokémon is active on
both sides at once**. This is singles with substitutions, not a double battle.

| RL team's Pokémon | Main strength | Main weakness |
|---|---|---|
| Swampert | Ground attacks beat Magnezone | Grass attacks from Meowscarada |
| Corviknight | Flying attacks beat Meowscarada | Electric attacks from Magnezone |

The scripted opponent brings Meowscarada and Magnezone. Both sides' leads are
randomly chosen by alternating the order of the two fixed Pokémon, so all four
starting matchups occur across training/evaluation. Switching is legal.

The scripted opponent switches to a counter in an unfavorable matchup with a
65% probability when one is available; otherwise it chooses a sensible
attacking move. The default **mixed** training opponent chooses *per battle*:
35% completely random policy, 65% scripted tactical policy. (Thus it is not
just the RandomPlayer or MaxBasePowerPlayer used in the first prototype.)

The RL observations contain own team's species, HP, active status (in exact
switch-index order), current *visible* opponent species and HP, boosts and
battle turn. Nothing about unrevealed opponents is copied into the input.

The underlying PPO and GAE implementation is imported from **your original**
`prototype.py`. The agent uses a new 17-feature network, and learns from scratch.
Do **not** try to load the 1v1 model: it expects seven features.

## Install on Windows / PyCharm

1. Extract this zip so you have the `pokemon_2v2_switching_addon` folder.
2. Copy `prototype_2v2.py` into the **same folder as** the original `prototype.py`.
3. In your terminal running Showdown, press `Ctrl+C` to **stop** it.
4. Replace `pokemon-showdown/config/custom-formats.ts` with this addon's
   `custom-formats.ts`. The new file preserves the old 1v1 format and adds the
   new `[Gen 9] RL 2v2 Singles` format. If you have since added other custom
   formats, merge the new entry instead of replacing your changes.
5. Restart Showdown from its directory:

   ```powershell
   cd pokemon-showdown
   node pokemon-showdown start --no-security
   ```

   Keep the server terminal open. A `--no-security` server should be kept
   strictly local/private and not exposed to the public Internet.
6. Open a **new** PyCharm terminal in `pokemon_1v1_rl_starter`. There is no need
   to reinstall Python dependencies. Smoke test:

   ```powershell
   .\.venv\Scripts\python.exe prototype_2v2.py smoke --battles 30
   ```

   You should see four lead matchups in `leads` (not necessarily all in only
   30 battles), a random-agent win rate, switch counts, and decisions/sec.
7. If successful, run a tiny PPO integration test:

   ```powershell
   .\.venv\Scripts\python.exe prototype_2v2.py train --steps 2000 --eval-every 1000 --eval-battles 30
   ```

8. Then do a more meaningful test (expect this to be harder/slower than 1v1):

   ```powershell
   .\.venv\Scripts\python.exe prototype_2v2.py train --steps 20000 --eval-every 5000 --eval-battles 100
   ```

   Training **starts from scratch** with this command; it overwrites
   `results_2v2/policy.pt` and `results_2v2/results.csv` only. Your earlier
   `results/policy.pt` is left untouched.

## What the outputs mean

- `vs random`: win rate against a completely random opponent in this new
  two-Pokémon domain.
- `vs tactical`: win rate against an opponent that attacks according to the
  matchup and sometimes makes counter-switches.
- `switches`: voluntary switches per battle; *forced* switches after a faint
  are excluded.
- `disadvantage_switch_rate` (in CSV): the fraction of decisions with an
  available switch and move while facing the designed type-disadvantage
  matchups where the agent voluntarily switches. **A higher rate is not
  automatically better:** sometimes staying in to attack is the right play.
- `train decisions/s`: Python PPO training throughput, excluding evaluation.
- `truncated_battles`: hitting the 70-turn safeguard; ideally very rare.

Consider this a **technical feasibility study**, not final experimental
results. The observations are tailored to these four Pokémon; a larger
roster will need a more general species/type/move representation. Learning
against even a decent scripted opponent is harder than the earlier 1v1 mirror.

## If it errors

- `Unknown format` or `invalid team`: verify the new `custom-formats.ts` was
  copied into Showdown and the server fully restarted. Check Showdown's
  terminal for the actual validation error.
- `ModuleNotFoundError: prototype`: `prototype_2v2.py` is not alongside your
  original `prototype.py`.
- Battle freezes: check both terminals and note the last turn/output. The
  script includes a 70-turn limit for endless-switching episodes.

## Sources / dependencies

- `poke-env` 0.16.1 `SinglesEnv`, `SingleAgentWrapper`, `Player.update_team`:
  https://github.com/hsahovic/poke-env/tree/0.16.1/src/poke_env
- Showdown custom rules:
  https://github.com/smogon/pokemon-showdown/blob/master/config/CUSTOM-RULES.md

Python syntax has been checked. Actual live integration needs to be verified
on your already-working local Showdown installation; this code is a new
experiment and has not yet run a live battle in the author's environment.
