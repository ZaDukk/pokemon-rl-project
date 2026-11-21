"""Curated Generation 9 Pokemon pool. Both sides sample 3 without replacement.

Fixed, documented sets prevent the team builder from silently changing the task.
All twelve are intended to be Gen 9 legal. Validate the full pool with Showdown
before training; Showdown is the authority for your installed simulator version.
"""
from __future__ import annotations
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

SPECIES = tuple(SETS)
NORMALIZE = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
MOVE_IDS = tuple(dict.fromkeys(
    NORMALIZE(line[2:]) for team in SETS.values()
    for line in team.splitlines() if line.startswith("- ")
))
ITEMS = tuple(dict.fromkeys(
    NORMALIZE(line.split(":", 1)[-1] if ":" in line else line.split("@",1)[1])
    for team in SETS.values() for line in team.splitlines() if " @ " in line
))
ABILITIES = tuple(dict.fromkeys(
    NORMALIZE(line.split(":", 1)[1])
    for team in SETS.values() for line in team.splitlines() if line.startswith("Ability:")
))


def sample_team(rng: random.Random | None = None, *, names=None) -> tuple[str, str, str]:
    """Return ordered species (first one is the lead)."""
    rng = rng or random
    if names is None:
        names = rng.sample(list(SPECIES), 3)
    names = tuple(names)
    if len(names) != 3 or len(set(names)) != 3 or any(s not in SETS for s in names):
        raise ValueError(f"Expected 3 distinct Pokemon from pool; got {names}")
    return names


def pack_team(names: tuple[str,str,str]) -> str:
    return "\n\n".join(SETS[s] for s in names)


def static_checks():
    assert len(SETS) == 12 and len(set(SPECIES)) == 12
    assert len(MOVE_IDS) >= 20
    for name, team in SETS.items():
        assert len([line for line in team.splitlines() if line.startswith("- ")]) == 4, name
    return {"species":len(SPECIES),"moves":len(MOVE_IDS),"items":len(ITEMS),"abilities":len(ABILITIES)}

if __name__ == '__main__':
    print(static_checks())
