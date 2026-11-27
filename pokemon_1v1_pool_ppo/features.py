"""Visible-state 1v1 observation encoder (adapted from the 3v3 encoder with TEAM_SIZE=1).

In 1v1 both Pokemon are revealed at turn 1, but opponent moves, item, ability
and hidden stats are still only encoded once Showdown reveals them."""
from __future__ import annotations
import math
import random
import numpy as np
import torch
from poke_env.battle import AbstractBattle
from pool import SPECIES, ABILITIES, ITEMS, MOVE_IDS

TEAM_SIZE = 1

TYPES = ("BUG", "DARK", "DRAGON", "ELECTRIC", "FAIRY", "FIGHTING", "FIRE",
         "FLYING", "GHOST", "GRASS", "GROUND", "ICE", "NORMAL", "POISON",
         "PSYCHIC", "ROCK", "STEEL", "WATER")
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
STATUSES = ("BRN", "FRZ", "PAR", "PSN", "SLP", "TOX", "FNT")
CATEGORIES = ("PHYSICAL", "SPECIAL", "STATUS")
SIDE_CONDS = ("STEALTH_ROCK", "SPIKES", "TOXIC_SPIKES", "STICKY_WEB",
              "REFLECT", "LIGHT_SCREEN", "AURORA_VEIL", "TAILWIND",
              "SAFEGUARD", "MIST")
WEATHERS = ("SUNNYDAY", "RAINDANCE", "SANDSTORM", "HAIL", "SNOW", "SNOWSCAPE")
FIELDS = ("ELECTRIC_TERRAIN", "GRASSY_TERRAIN", "MISTY_TERRAIN",
          "PSYCHIC_TERRAIN", "TRICK_ROOM", "MAGIC_ROOM", "WONDER_ROOM")
POKEMON_EFFECTS = ("PROTECT", "LEECH_SEED", "SUBSTITUTE", "CONFUSION",
                   "TAUNT", "ENCORE", "YAWN", "FLINCH", "TRAPPED")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # Faster for these small CPU models.


def enum_name(x) -> str:
    return str(getattr(x, "name", "")).upper() if x is not None else ""


def sid(x) -> str:
    # Stable normalized IDs for Pokemon / abilities / items / moves.
    return "".join(ch for ch in str(x or "").lower() if ch.isalnum())


def species(mon) -> str:
    return sid(mon.species) if mon is not None else ""


def onehot(value: str, vocab: tuple[str, ...]) -> list[float]:
    # Adds a dedicated UNKNOWN / ABSENT bin, rather than silently mapping to
    # an unrelated category. For type2 None this is also the 'no type2' bin.
    v = [0.0] * (len(vocab) + 1)
    v[vocab.index(value) if value in vocab else -1] = 1.0
    return v


def bounded(value, scale: float = 1.0, low: float = 0.0, high: float = 1.0) -> float:
    try:
        result = float(value) / scale
    except (TypeError, ValueError):
        return 0.0
    return float(np.clip(result, low, high)) if math.isfinite(result) else 0.0


def hp(mon) -> float:
    return bounded(mon.current_hp_fraction) if mon is not None else 0.0


def bad_matchup(battle) -> bool:
    """Public type-based danger heuristic, not an optimal-move oracle."""
    a, b = battle.active_pokemon, battle.opponent_active_pokemon
    if a is None or b is None:
        return False
    own_danger = max((matchup_type_multiplier(t, a) for t in (b.type_1, b.type_2) if t is not None), default=0)
    our_offense = max((matchup_type_multiplier(t, b) for t in (a.type_1, a.type_2) if t is not None), default=0)
    return own_danger >= 0.5 and our_offense <= 0.25


def matchup_type_multiplier(attack_type, defender) -> float:
    if attack_type is None or defender is None:
        return 0.0  # UNKNOWN; separate presence features disambiguate.
    try:
        return bounded(defender.damage_multiplier(attack_type), 4.0)
    except (ValueError, KeyError):
        return 0.0


def move_features(move, attacker=None, defender=None) -> list[float]:
    """A move descriptor. Unknown moves have an all-zero representation.

    The first four slots of our active Pokemon are in EXACTLY the order of
    active_pokemon.moves.values(), matching SinglesEnv actions 6..9.
    Opponent's observed move slots are sorted by move ID; their order has no
    connection to our action indices.
    """
    if move is None:
        # Populate the same shape, with all zeros.
        return [0.0] * MOVE_WIDTH
    mid = sid(move.id)
    move_type = enum_name(move.type)
    cat = enum_name(move.category)
    sec = getattr(move, "secondary", None) or []
    boost_own = getattr(move, "self_boost", None) or {}
    boost_target = getattr(move, "boosts", None) or {}
    known_attacker_types = (attacker.type_1, attacker.type_2) if attacker else (None, None)
    feats = (
        [1.0]
        + onehot(mid, MOVE_IDS)
        + onehot(move_type, TYPES)
        + onehot(cat, CATEGORIES)
        + [
            bounded(move.base_power, 250),
            bounded(move.accuracy),
            bounded(getattr(move, "current_pp", 0), max(int(move.max_pp), 1)),
            bounded(move.max_pp, 64),
            bounded(move.priority, 7, -1, 1),
            bounded(getattr(move, "expected_hits", 1), 5),
            bounded(getattr(move, "crit_ratio", 0), 6),
            bounded(getattr(move, "heal", 0)),
            bounded(getattr(move, "drain", 0)),
            bounded(getattr(move, "recoil", 0)),
            float(bool(getattr(move, "self_switch", False))),
            float(bool(getattr(move, "is_protect_move", False))),
            float(bool(getattr(move, "is_last_used", False))),
            float(move.type in known_attacker_types),  # STAB approximation
            matchup_type_multiplier(move.type, defender),
            float(bool(sec)),
            bounded(sum(max(0, int(v)) for v in boost_own.values()), 6),
            bounded(sum(max(0, int(v)) for v in boost_target.values()), 6),
            float(getattr(move, "status", None) is not None),
        ]
    )
    return feats


# Calculate the width directly from the representation specification, avoiding
# a dummy poke-env object or a hidden dependency on battle initialization.
MOVE_WIDTH = (1 + (len(MOVE_IDS) + 1) + (len(TYPES) + 1)
              + (len(CATEGORIES) + 1) + 19)


def known_moves(mon, *, owned: bool) -> list:
    if mon is None:
        return []
    moves = list(mon.moves.values())
    return moves[:4] if owned else sorted(moves, key=lambda m: m.id)[:4]


def pokemon_features(mon, *, owned: bool, other_active=None) -> list[float]:
    """Encode only information visible from this player's point of view.

    Public species/base stats/types are available once opponent is revealed;
    opponent EVs, IVs and raw stats are deliberately not used.
    Opponent moves are only those already observed by Showdown/poke-env.
    """
    if mon is None:
        return [0.0] * POKEMON_WIDTH

    bs = mon.base_stats or {}
    raw_stats = mon.stats or {} if owned else {}  # Never expose hidden opponent spreads
    ab = sid(getattr(mon, "ability", None))
    it = sid(getattr(mon, "item", None))
    own_types = [enum_name(mon.type_1), enum_name(mon.type_2)]
    effect_names = {enum_name(e) for e in (getattr(mon, "effects", {}) or {})}
    known_move_list = known_moves(mon, owned=owned)
    effect_to_foe = [
        matchup_type_multiplier(mon.type_1, other_active),
        matchup_type_multiplier(mon.type_2, other_active),
    ]
    # A defensive matchup proxy from publicly known opponent types; it is NOT
    # an exact damage forecast and does not assume unrevealed opponent moves.
    foe_to_us = [
        matchup_type_multiplier(other_active.type_1, mon) if other_active else 0.0,
        matchup_type_multiplier(other_active.type_2, mon) if other_active else 0.0,
    ]
    feats = (
        [1.0, float(bool(mon.active)), float(bool(mon.fainted)),
         float(bool(mon.revealed)) if not owned else 1.0,
         hp(mon), bounded(mon.level, 100),
         float(bool(getattr(mon, "first_turn", False))),
         bounded(getattr(mon, "protect_counter", 0), 6),
         float(bool(getattr(mon, "must_recharge", False))),
         float(bool(getattr(mon, "is_terastallized", False)))]
        + onehot(species(mon), SPECIES)
        + onehot(own_types[0], TYPES)
        + onehot(own_types[1], TYPES)
        + [bounded(bs.get(k), 255) for k in STAT_KEYS]
        + [bounded(raw_stats.get(k), 650) for k in STAT_KEYS]
        + [float(raw_stats.get(k) is not None) for k in STAT_KEYS]
        + [bounded(mon.boosts.get(k, 0), 6, -1, 1) for k in BOOST_KEYS]
        + onehot(enum_name(mon.status), STATUSES)
        + [bounded(getattr(mon, "status_counter", 0), 7)]
        + onehot(ab, ABILITIES)
        + onehot(it, ITEMS)
        + [float(k in effect_names) for k in POKEMON_EFFECTS]
        + effect_to_foe + foe_to_us
        + [bounded(len(known_move_list), 4)]
    )
    for m in (known_move_list + [None] * 4)[:4]:
        feats.extend(move_features(m, attacker=mon, defender=other_active))
    if len(feats) != POKEMON_WIDTH:
        raise AssertionError(f"Pokemon encoding length {len(feats)} != {POKEMON_WIDTH}")
    return feats


POKEMON_WIDTH = (10 + (len(SPECIES) + 1) + 2 * (len(TYPES) + 1)
                 + 6 + 6 + 6 + len(BOOST_KEYS) + (len(STATUSES) + 1)
                 + 1 + (len(ABILITIES) + 1) + (len(ITEMS) + 1)
                 + len(POKEMON_EFFECTS) + 4 + 1 + 4 * MOVE_WIDTH)


def conditions_vector(conds: dict | None) -> list[float]:
    names = {enum_name(c): val for c, val in (conds or {}).items()}
    # Non-stacking conditions are represented by presence, stacking hazards
    # by number of layers, not their 'turn created' counter.
    return [bounded(names.get(k, 0), 3) if k in {"SPIKES", "TOXIC_SPIKES"}
            else float(k in names) for k in SIDE_CONDS]


def battle_features(battle: AbstractBattle | None) -> np.ndarray:
    own = list(battle.team.values())[:TEAM_SIZE] if battle is not None else []
    foe = list(battle.opponent_team.values()) if battle is not None else []
    foe_active = battle.opponent_active_pokemon if battle is not None else None
    our_active = battle.active_pokemon if battle is not None else None
    # On ordinary battles opponent_team only contains revealed opponents.
    # Their ordering is not our action ordering, so use a stable species sort.
    if foe_active is not None and all(p is not foe_active for p in foe):
        foe.append(foe_active)
    foe = sorted(foe, key=species)[:TEAM_SIZE]
    known_foe_count = len(foe)
    our_active_moves = known_moves(our_active, owned=True)
    legal_switch_count = len(battle.available_switches) if battle is not None else 0
    legal_move_count = len(battle.available_moves) if battle is not None else 0
    fe = [
        bounded(battle.turn, 160) if battle is not None else 0.0,
        bounded(sum(not p.fainted for p in own), TEAM_SIZE),
        bounded(sum(p.fainted for p in foe), TEAM_SIZE),
        bounded(known_foe_count, TEAM_SIZE),
        bounded(legal_switch_count, TEAM_SIZE),
        bounded(legal_move_count, 4),
        float(bool(battle.trapped)) if battle is not None else 0.0,
        float(bool(battle.can_tera)) if battle is not None else 0.0,
        float(bool(battle.used_tera)) if battle is not None else 0.0,
        float(bool(battle.opponent_used_tera)) if battle is not None else 0.0,
        bounded(len(our_active_moves), 4),
    ]
    weather = {enum_name(x) for x in (battle.weather if battle else {})}
    fields = {enum_name(x) for x in (battle.fields if battle else {})}
    fe += [float(k in weather) for k in WEATHERS]
    fe += [float(k in fields) for k in FIELDS]
    fe += conditions_vector(battle.side_conditions if battle else {})
    fe += conditions_vector(battle.opponent_side_conditions if battle else {})

    # Own team ordering MUST match switch actions 0..5.
    for p in (own + [None] * TEAM_SIZE)[:TEAM_SIZE]:
        fe += pokemon_features(p, owned=True, other_active=foe_active)
    # Observed opponent, including revealed moves and known HP/status only.
    for p in (foe + [None] * TEAM_SIZE)[:TEAM_SIZE]:
        fe += pokemon_features(p, owned=False, other_active=our_active)

    # Action-specific features reinforce that switches 0/1 correspond to own
    # team slots and moves 6..9 correspond to own active's known move slots.
    # These features overlap Pokemon blocks deliberately (easier for an MLP).
    for p in (own + [None] * TEAM_SIZE)[:TEAM_SIZE]:
        if p is None:
            fe += [0.0] * 7
        else:
            p_moves = known_moves(p, owned=True)
            effectiveness = [matchup_type_multiplier(m.type, foe_active) for m in p_moves]
            fe += [
                float(bool(p.active)), float(bool(p.fainted)), hp(p),
                max(effectiveness, default=0.0),
                matchup_type_multiplier(foe_active.type_1, p) if foe_active else 0.0,
                matchup_type_multiplier(foe_active.type_2, p) if foe_active else 0.0,
                bounded(p.base_stats.get("spe", 0), 255),
            ]
    if battle is None:
        fe += [0.0] * 4
    else:
        known_ids = {m.id for m in battle.available_moves}
        fe += [float(m is not None and m.id in known_ids)
               for m in (our_active_moves + [None] * 4)[:4]]
    arr = np.asarray(fe, dtype=np.float32)
    if arr.shape != (N_FEATURES,):
        raise AssertionError(f"Observation shape {arr.shape}, expected ({N_FEATURES},)")
    if not np.isfinite(arr).all() or (np.abs(arr) > 1.00001).any():
        raise ValueError("Invalid observation: NaN/inf or out-of-range feature")
    return arr


GLOBAL_WIDTH = 11 + len(WEATHERS) + len(FIELDS) + 2 * len(SIDE_CONDS)
N_FEATURES = GLOBAL_WIDTH + 2 * TEAM_SIZE * POKEMON_WIDTH + TEAM_SIZE * 7 + 4


