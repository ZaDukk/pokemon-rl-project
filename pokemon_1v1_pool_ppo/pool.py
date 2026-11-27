"""Pokemon pool for the 1v1 experiment: each side gets ONE Pokemon drawn from a
small fixed pool (default 6 of the 12 documented sets below).

The 12 sets are copied verbatim from ../pokemon_3v3_pool_ppo/pool.py so that the
two experiments use identical Pokemon. The *feature vocabulary* always covers all
12 sets (so the observation size never changes); the *active pool* is the subset
actually sampled in battles and is chosen on the command line with --pool.
Showdown is the authority on legality: run `check-pool` before training.
"""
from __future__ import annotations
import itertools
import random
import re

SETS = {
    "swampert": """Swampert @ Leftovers
Ability: Torrent
Level: 100
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Earthquake
- Waterfall
- Ice Punch
- Protect""",
    "corviknight": """Corviknight @ Leftovers
Ability: Pressure
Level: 100
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Brave Bird
- Body Press
- Roost
- U-turn""",
    "meowscarada": """Meowscarada @ Life Orb
Ability: Overgrow
Level: 100
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Flower Trick
- Knock Off
- U-turn
- Play Rough""",
    "magnezone": """Magnezone @ Leftovers
Ability: Sturdy
Level: 100
EVs: 252 HP / 252 SpA / 4 SpD
Modest Nature
- Thunderbolt
- Flash Cannon
- Volt Switch
- Thunder Wave""",
    "garchomp": """Garchomp @ Rocky Helmet
Ability: Rough Skin
Level: 100
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Earthquake
- Dragon Claw
- Stone Edge
- Swords Dance""",
    "rotomwash": """Rotom-Wash @ Leftovers
Ability: Levitate
Level: 100
EVs: 252 HP / 4 SpA / 252 Spe
Timid Nature
- Hydro Pump
- Volt Switch
- Thunderbolt
- Will-O-Wisp""",
    "skeledirge": """Skeledirge @ Heavy-Duty Boots
Ability: Unaware
Level: 100
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Torch Song
- Shadow Ball
- Slack Off
- Will-O-Wisp""",
    "greattusk": """Great Tusk @ Leftovers
Ability: Protosynthesis
Level: 100
EVs: 252 HP / 252 Atk / 4 Def
Adamant Nature
- Headlong Rush
- Close Combat
- Knock Off
- Rapid Spin""",
    "dragapult": """Dragapult @ Life Orb
Ability: Infiltrator
Level: 100
EVs: 4 Atk / 252 SpA / 252 Spe
Hasty Nature
- Draco Meteor
- Shadow Ball
- Flamethrower
- U-turn""",
    "primarina": """Primarina @ Leftovers
Ability: Torrent
Level: 100
EVs: 252 HP / 252 SpA / 4 SpD
Modest Nature
- Moonblast
- Surf
- Ice Beam
- Calm Mind""",
    "amoonguss": """Amoonguss @ Rocky Helmet
Ability: Regenerator
Level: 100
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Giga Drain
- Sludge Bomb
- Spore
- Synthesis""",
    "kingambit": """Kingambit @ Black Glasses
Ability: Supreme Overlord
Level: 100
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Kowtow Cleave
- Iron Head
- Sucker Punch
- Swords Dance""",
}

SPECIES = tuple(SETS)  # Full vocabulary (12). Do NOT shrink: features depend on it.
NORMALIZE = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
MOVE_IDS = tuple(dict.fromkeys(
    NORMALIZE(line[2:]) for team in SETS.values()
    for line in team.splitlines() if line.startswith("- ")
))
ITEMS = tuple(dict.fromkeys(
    NORMALIZE(line.split("@", 1)[1])
    for team in SETS.values() for line in team.splitlines() if " @ " in line
))
ABILITIES = tuple(dict.fromkeys(
    NORMALIZE(line.split(":", 1)[1])
    for team in SETS.values() for line in team.splitlines() if line.startswith("Ability:")
))

# Default 6-Pokemon pool: a Fire/Water/Grass triangle (Skeledirge, Primarina,
# Meowscarada) plus Swampert, Garchomp and Rotom-Wash, so that type matchups
# matter but no single species is immune to everything. Change with --pool.
DEFAULT_POOL = ("garchomp", "swampert", "meowscarada", "skeledirge", "rotomwash", "primarina")


def parse_pool(text: str | None = None, size: int | None = None) -> tuple[str, ...]:
    """Parse 'garchomp,swampert,...' (or use DEFAULT_POOL), optionally truncated.

    --pool-size N with no --pool takes the first N of DEFAULT_POOL then the rest
    of the 12 sets, so pool sizes 2..12 are nested and comparable.
    """
    if text:
        names = tuple(NORMALIZE(x) for x in text.split(",") if x.strip())
    else:
        names = DEFAULT_POOL + tuple(s for s in SPECIES if s not in DEFAULT_POOL)
        names = names[: (size or len(DEFAULT_POOL))]
    if size is not None and text:
        names = names[:size]
    bad = [n for n in names if n not in SETS]
    if bad:
        raise ValueError(f"Unknown Pokemon {bad}; choose from {list(SPECIES)}")
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError(f"Pool needs >=2 distinct Pokemon; got {names}")
    return names


def sample_mon(pool, rng: random.Random | None = None, *, name: str | None = None) -> str:
    rng = rng or random
    name = NORMALIZE(name) if name else rng.choice(list(pool))
    if name not in SETS:
        raise ValueError(f"{name} is not one of the 12 documented sets")
    return name


def sample_matchup(pool, rng: random.Random | None = None, *, mirrors: bool = True):
    """Independent uniform draw for each side; mirrors (X vs X) allowed by default."""
    rng = rng or random
    ours = rng.choice(list(pool))
    choices = list(pool) if mirrors else [p for p in pool if p != ours]
    return ours, rng.choice(choices)


def all_matchups(pool, *, mirrors: bool = True):
    """Every ordered (ours, theirs) pair: 36 for a pool of 6 with mirrors."""
    return [(a, b) for a, b in itertools.product(pool, pool) if mirrors or a != b]


def pack_team(name: str) -> str:
    return SETS[name]


def static_checks():
    assert len(SETS) == 12 and len(set(SPECIES)) == 12
    for name, team in SETS.items():
        assert len([l for l in team.splitlines() if l.startswith("- ")]) == 4, name
    assert set(DEFAULT_POOL) <= set(SETS)
    return {"vocab_species": len(SPECIES), "moves": len(MOVE_IDS), "items": len(ITEMS),
            "abilities": len(ABILITIES), "default_pool": list(DEFAULT_POOL)}


if __name__ == "__main__":
    print(static_checks())
