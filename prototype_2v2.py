"""Information-rich 2v2 Pokemon Showdown (singles) with SELF-IMPLEMENTED masked PPO.

Designed for poke-env==0.16.1, the existing 'gen9rl2v2singles' custom
format, and the same four fixed Pokemon as the earlier prototype.

No stable-baselines3 / imported RL algorithm is used. poke-env supplies the
battle simulator, legal-action mask and battle observations only.

Run (from your original starter project directory):
    python prototype_2v2.py smoke --battles 20
    python prototype_2v2.py train --steps 100000 --eval-every 5000 --eval-battles 200
    python prototype_2v2.py evaluate --weights results_2v2_rich/policy.pt --battles 400

IMPORTANT: old 17-feature policy.pt weights are NOT compatible. New results
are written to results_2v2_rich by default; the old models are untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import RandomPlayer
from torch import nn
from torch.distributions import Categorical

# ------------------------------- Format ----------------------------------
FORMAT = "gen9rl2v2singles"
MAX_TEAM = 2
ACTION_COUNT = 26  # 6 switches, 4 moves and Gen-9 gimmick action slots.
MAX_TURNS_WARNING = 160  # ERROR if exceeded; NEVER pretend an active battle ended.

SWAMPERT = """Swampert @ Leftovers
Ability: Torrent
Level: 100
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Earthquake
- Waterfall
- Ice Punch
- Protect"""
CORVIKNIGHT = """Corviknight @ Leftovers
Ability: Pressure
Level: 100
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Brave Bird
- Body Press
- Roost
- U-turn"""
MEOWSCARADA = """Meowscarada @ Life Orb
Ability: Overgrow
Level: 100
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Flower Trick
- Knock Off
- U-turn
- Play Rough"""
MAGNEZONE = """Magnezone @ Leftovers
Ability: Sturdy
Level: 100
EVs: 252 HP / 252 SpA / 4 SpD
Modest Nature
- Thunderbolt
- Flash Cannon
- Volt Switch
- Thunder Wave"""

MY_TEAMS = [SWAMPERT + "\n\n" + CORVIKNIGHT,
            CORVIKNIGHT + "\n\n" + SWAMPERT]
OPP_TEAMS = [MEOWSCARADA + "\n\n" + MAGNEZONE,
             MAGNEZONE + "\n\n" + MEOWSCARADA]
MY_SPECIES = ("swampert", "corviknight")
OPP_SPECIES = ("meowscarada", "magnezone")

# Dictionaries are FIXED before training. All choices and missing-value slots
# have consistent numerical meaning throughout training and evaluation.
SPECIES = ("swampert", "corviknight", "meowscarada", "magnezone")
TYPES = ("BUG", "DARK", "DRAGON", "ELECTRIC", "FAIRY", "FIGHTING", "FIRE",
         "FLYING", "GHOST", "GRASS", "GROUND", "ICE", "NORMAL", "POISON",
         "PSYCHIC", "ROCK", "STEEL", "WATER")
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
STATUSES = ("BRN", "FRZ", "PAR", "PSN", "SLP", "TOX", "FNT")
ABILITIES = ("torrent", "pressure", "overgrow", "sturdy")
ITEMS = ("leftovers", "lifeorb")
MOVE_IDS = ("earthquake", "waterfall", "icepunch", "protect", "bravebird",
            "bodypress", "roost", "uturn", "flowertrick", "knockoff",
            "playrough", "thunderbolt", "flashcannon", "voltswitch",
            "thunderwave")
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
    return (species(battle.active_pokemon), species(battle.opponent_active_pokemon)) in {
        ("swampert", "meowscarada"), ("corviknight", "magnezone"),
    }


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
    own = list(battle.team.values())[:MAX_TEAM] if battle is not None else []
    foe = list(battle.opponent_team.values()) if battle is not None else []
    foe_active = battle.opponent_active_pokemon if battle is not None else None
    our_active = battle.active_pokemon if battle is not None else None
    # On ordinary battles opponent_team only contains revealed opponents.
    # Their ordering is not our action ordering, so use a stable species sort.
    if foe_active is not None and all(p is not foe_active for p in foe):
        foe.append(foe_active)
    foe = sorted(foe, key=species)[:MAX_TEAM]
    known_foe_count = len(foe)
    our_active_moves = known_moves(our_active, owned=True)
    legal_switch_count = len(battle.available_switches) if battle is not None else 0
    legal_move_count = len(battle.available_moves) if battle is not None else 0
    fe = [
        bounded(battle.turn, 160) if battle is not None else 0.0,
        bounded(sum(not p.fainted for p in own), MAX_TEAM),
        bounded(sum(p.fainted for p in foe), MAX_TEAM),
        bounded(known_foe_count, MAX_TEAM),
        bounded(legal_switch_count, MAX_TEAM),
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
    for p in (own + [None] * MAX_TEAM)[:MAX_TEAM]:
        fe += pokemon_features(p, owned=True, other_active=foe_active)
    # Observed opponent, including revealed moves and known HP/status only.
    for p in (foe + [None] * MAX_TEAM)[:MAX_TEAM]:
        fe += pokemon_features(p, owned=False, other_active=our_active)

    # Action-specific features reinforce that switches 0/1 correspond to own
    # team slots and moves 6..9 correspond to own active's known move slots.
    # These features overlap Pokemon blocks deliberately (easier for an MLP).
    for p in (own + [None] * MAX_TEAM)[:MAX_TEAM]:
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
N_FEATURES = GLOBAL_WIDTH + 4 * POKEMON_WIDTH + 2 * 7 + 4


class RichTwoVsTwoEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(low=-1.0, high=1.0, shape=(N_FEATURES,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return battle_features(battle)

    def calc_reward(self, battle: AbstractBattle) -> float:
        # EXACTLY same reward as 17-feature prototype; isolates the effect of
        # changing observation space (same 2v2 teams and opponent policies).
        return self.reward_computing_helper(
            battle, number_of_pokemons=MAX_TEAM,
            hp_value=1.0, fainted_value=1.0, victory_value=5.0,
        )


class TacticalOpponent(RandomPlayer):
    """Same opponent logic as original 2v2 experiment, for comparability."""
    def __init__(self, *, mode="tactical", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self._random_battles = {}

    def choose_move(self, battle):
        if self.mode == "mixed":
            tag = battle.battle_tag
            if tag not in self._random_battles:
                self._random_battles[tag] = random.random() < 0.35
                if len(self._random_battles) > 5000:
                    self._random_battles = {tag: self._random_battles[tag]}
            if self._random_battles[tag]:
                return self.choose_random_move(battle)
        own, foe = species(battle.active_pokemon), species(battle.opponent_active_pokemon)
        if battle.active_pokemon is None or battle.active_pokemon.fainted:
            if battle.available_switches:
                return self.create_order(battle.available_switches[0])
            return self.choose_random_move(battle)
        target_type = ("meowscarada" if own == "magnezone" and foe == "swampert"
                       else "magnezone" if own == "meowscarada" and foe == "corviknight"
                       else "")
        if target_type and not battle.trapped and random.random() < 0.65:
            mon = next((p for p in battle.available_switches
                        if species(p) == target_type), None)
            if mon is not None:
                return self.create_order(mon)
        preferred = ("flowertrick" if own == "meowscarada" and foe == "swampert"
                     else "thunderbolt" if own == "magnezone" and foe == "corviknight"
                     else "knockoff" if own == "meowscarada"
                     else "flashcannon" if own == "magnezone"
                     else "")
        chosen = next((m for m in battle.available_moves if m.id == preferred), None)
        if chosen is not None:
            return self.create_order(chosen)
        if battle.available_moves:
            return self.create_order(max(battle.available_moves, key=lambda m: float(m.base_power or 0)))
        return self.choose_random_move(battle)


def make_env(opponent="mixed") -> SingleAgentWrapper:
    env = RichTwoVsTwoEnv(
        battle_format=FORMAT, team=MY_TEAMS[0],
        choose_on_teampreview=False, log_level=40,
        open_timeout=10.0, challenge_timeout=45.0,
    )
    env.agent2.update_team(OPP_TEAMS[0])
    if opponent == "random":
        agent = RandomPlayer(battle_format=FORMAT, start_listening=False)
    else:
        agent = TacticalOpponent(mode=opponent, battle_format=FORMAT, start_listening=False)
    return SingleAgentWrapper(env, agent)


def reset_with_leads(env, ours: int | None = None, theirs: int | None = None):
    # This is supported poke-env public update_team, not hidden-state access.
    ours = random.randrange(2) if ours is None else ours
    theirs = random.randrange(2) if theirs is None else theirs
    env.env.agent1.update_team(MY_TEAMS[ours])
    env.env.agent2.update_team(OPP_TEAMS[theirs])
    return env.reset()


# ---------------------- Self-implemented PPO ------------------------------
class ActorCritic2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_FEATURES, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.policy = nn.Linear(256, ACTION_COUNT)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if not mask.any(dim=-1).all():
            raise ValueError("No legal actions in action mask")
        z = self.encoder(x)
        logits = self.policy(z).masked_fill(~mask.bool(), -1e9)
        return Categorical(logits=logits), self.value_head(z).squeeze(-1)


def as_tensors(obs: dict):
    x = torch.as_tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
    mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0)
    if x.shape != (1, N_FEATURES) or mask.shape != (1, ACTION_COUNT):
        raise ValueError(f"Unexpected observation/mask shape: {x.shape}, {mask.shape}")
    return x, mask


@torch.no_grad()
def choose_action(model: ActorCritic2 | None, observation, deterministic=False):
    if model is None:
        choices = np.flatnonzero(observation["action_mask"])
        if len(choices) == 0:
            raise ValueError("No legal actions")
        return int(np.random.choice(choices)), 0.0, 0.0
    state, mask = as_tensors(observation)
    dist, value = model(state, mask)
    action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
    return int(action.item()), float(dist.log_prob(action).item()), float(value.item())


def ppo_update(model, optimizer, batch, bootstrap, *, gamma=.99, gae_lambda=.95,
               clip=.2, epochs=4, minibatch=64, entropy_coef=.02,
               value_coef=.5):
    """Our own clipped PPO and GAE implementation, no third-party RL trainer."""
    rewards = np.asarray(batch["reward"], np.float32)
    values = np.asarray(batch["value"], np.float32)
    dones = np.asarray(batch["done"], np.float32)
    adv = np.zeros_like(rewards)
    gae = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        next_value = bootstrap if i == len(rewards) - 1 else values[i + 1]
        is_live = 1.0 - dones[i]
        delta = rewards[i] + gamma * next_value * is_live - values[i]
        gae = delta + gamma * gae_lambda * is_live * gae
        adv[i] = gae
    returns = adv + values

    obs = torch.as_tensor(np.stack(batch["obs"]), dtype=torch.float32)
    masks = torch.as_tensor(np.stack(batch["mask"]), dtype=torch.bool)
    actions = torch.as_tensor(batch["action"], dtype=torch.long)
    old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32)
    returns = torch.as_tensor(returns, dtype=torch.float32)
    adv = torch.as_tensor(adv, dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    metrics = defaultdict(float)
    batches = 0
    for _ in range(epochs):
        for indices in torch.randperm(len(rewards)).split(minibatch):
            dist, val = model(obs[indices], masks[indices])
            logp = dist.log_prob(actions[indices])
            ratio = (logp - old_logp[indices]).exp()
            policy_loss = -torch.minimum(ratio * adv[indices],
                                         ratio.clamp(1 - clip, 1 + clip) * adv[indices]).mean()
            value_loss = 0.5 * (returns[indices] - val).square().mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.item()
            metrics["approx_kl"] += ((ratio - 1) - (logp - old_logp[indices])).mean().item()
            batches += 1
    return {k: v / max(batches, 1) for k, v in metrics.items()}


def ensure_battle_progress(battle):
    if battle.turn >= MAX_TURNS_WARNING:
        # Crucially do NOT mark the episode as ended. Doing so without actually
        # finishing Showdown's battle can deadlock the next env.reset().
        raise RuntimeError(
            f"Battle reached {battle.turn} turns. The live Showdown battle "
            f"was NOT reset. Inspect switching loops / opponent policy."
        )


def evaluate(env, model, games: int, *, balanced=True, verbose=False):
    if games <= 0:
        raise ValueError("games must be positive")
    stats = defaultdict(float)
    per_lead = defaultdict(lambda: [0, 0])
    t0 = time.perf_counter()
    for i in range(games):
        ours, theirs = ((i // 2) % 2, i % 2) if balanced else (None, None)
        obs, _ = reset_with_leads(env, ours, theirs)
        b = env.env.battle1
        lead = f"{species(b.active_pokemon)} vs {species(b.opponent_active_pokemon)}"
        done = False
        while not done:
            b = env.env.battle1
            action, _, _ = choose_action(model, obs, deterministic=True)
            mask = np.asarray(obs["action_mask"], dtype=bool)
            can_switch = bool(mask[:6].any())
            can_attack = bool(mask[6:10].any())
            if bad_matchup(b) and can_switch and can_attack:
                stats["bad_opportunities"] += 1
                if action < 6:
                    stats["bad_switches"] += 1
            if action < 6:
                if can_attack:
                    stats["voluntary_switches"] += 1
                else:
                    stats["forced_switches"] += 1
            obs, _, terminated, truncated, _ = env.step(np.int64(action))
            stats["decisions"] += 1
            done = bool(terminated or truncated)
            if not done:
                ensure_battle_progress(env.env.battle1)
        b = env.env.battle1
        stats["wins"] += int(bool(b.won))
        stats["ties"] += int(not b.won and not b.lost)
        per_lead[lead][0] += int(bool(b.won))
        per_lead[lead][1] += 1
        if verbose and (i + 1) % max(1, games // 10) == 0:
            print(f"  evaluation {i+1}/{games} battles", flush=True)
    elapsed = time.perf_counter() - t0
    return {
        "win_rate": stats["wins"] / games,
        "tie_rate": stats["ties"] / games,
        "voluntary_switches_per_battle": stats["voluntary_switches"] / games,
        "forced_switches_per_battle": stats["forced_switches"] / games,
        "disadvantage_switch_rate": stats["bad_switches"] / max(1, stats["bad_opportunities"]),
        "bad_opportunities": int(stats["bad_opportunities"]),
        "decisions_per_second": stats["decisions"] / max(elapsed, 1e-9),
        "battles_per_second": games / max(elapsed, 1e-9),
        "mean_decisions": stats["decisions"] / games,
        "elapsed_seconds": elapsed,
        "lead_results": {k: {"wins": v[0], "games": v[1]} for k, v in per_lead.items()},
    }


def new_batch():
    return {k: [] for k in ("obs", "mask", "action", "logp", "value", "reward", "done")}


def write_metadata(out: Path, args):
    metadata = {
        "version": "rich-2v2-v1", "poke_env": "0.16.1",
        "format": FORMAT, "features": N_FEATURES, "action_count": ACTION_COUNT,
        "own_team": list(MY_SPECIES), "opponent_team": list(OPP_SPECIES),
        "reward": {"hp_value": 1.0, "fainted_value": 1.0, "victory_value": 5.0},
        "algorithm": "self-implemented clipped PPO with GAE and legal-action masking",
        "args": vars(args),
    }
    (out / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_checkpoint(out: Path, model, optimizer, steps, battles, *, rolling=True):
    torch.save(model.state_dict(), out / f"policy_{steps:07d}.pt")
    if rolling:
        torch.save(model.state_dict(), out / "policy.pt")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "steps": steps, "battles": battles,
            "py_random": random.getstate(),
            "np_random": np.random.get_state(),
            "torch_random": torch.get_rng_state(),
            "features": N_FEATURES,
        }, out / "training_checkpoint.pt")


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = ActorCritic2()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = battles = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt["features"] != N_FEATURES:
            raise ValueError("Cannot resume: feature dimensions differ")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        steps, battles = ckpt["steps"], ckpt["battles"]
        random.setstate(ckpt["py_random"])
        np.random.set_state(ckpt["np_random"])
        torch.set_rng_state(ckpt["torch_random"])
        print(f"Resuming model from {args.resume} at {steps} decisions. "
              "A new battle starts (live simulator state is not serialized).", flush=True)
    write_metadata(out, args)
    train_env = make_env(args.opponent)
    eval_random = make_env("random")
    eval_tactical = make_env("tactical")
    csv_path = out / "results.csv"
    columns = ["steps", "train_battles", "training_decisions_per_second",
               "vs_random_win_rate", "vs_tactical_win_rate",
               "vs_random_switches_per_battle", "vs_tactical_switches_per_battle",
               "vs_tactical_disadvantage_switch_rate", "vs_tactical_bad_opportunities",
               "tactical_swamp_vs_meow", "tactical_swamp_vs_mag",
               "tactical_corv_vs_meow", "tactical_corv_vs_mag",
               "ppo_policy_loss", "ppo_value_loss", "ppo_entropy", "ppo_approx_kl"]
    if steps == 0 or not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()
    start = time.perf_counter()
    active_training_seconds = 0.0
    next_evaluation = ((steps // args.eval_every) + 1) * args.eval_every
    try:
        print(f"Rich observation: {N_FEATURES} inputs; {sum(p.numel() for p in model.parameters()):,} "
              f"trainable parameters. Reward unchanged from 17-feature version.", flush=True)
        print("Untrained RANDOM-POLICY benchmarks:", flush=True)
        for label, e in (("random", eval_random), ("tactical", eval_tactical)):
            s = evaluate(e, None, args.eval_battles)
            print(f"  vs {label}: wins {s['win_rate']:.1%}; "
                  f"voluntary switches {s['voluntary_switches_per_battle']:.2f}/battle", flush=True)

        obs, _ = reset_with_leads(train_env)
        last_losses = {}
        while steps < args.steps:
            batch = new_batch()
            t0 = time.perf_counter()
            for _ in range(min(args.rollout, args.steps - steps)):
                action, logp, val = choose_action(model, obs)
                batch["obs"].append(obs["observation"].copy())
                batch["mask"].append(obs["action_mask"].copy())
                batch["action"].append(action)
                batch["logp"].append(logp)
                batch["value"].append(val)
                obs, reward, terminated, truncated, _ = train_env.step(np.int64(action))
                done = bool(terminated or truncated)
                batch["reward"].append(float(reward))
                batch["done"].append(done)
                steps += 1
                if done:
                    battles += 1
                    obs, _ = reset_with_leads(train_env)
                else:
                    ensure_battle_progress(train_env.env.battle1)
            if batch["done"][-1]:
                bootstrap = 0.0
            else:
                with torch.no_grad():
                    x, mask = as_tensors(obs)
                    _, val = model(x, mask)
                    bootstrap = float(val.item())
            last_losses = ppo_update(
                model, optimizer, batch, bootstrap, gamma=args.gamma,
                gae_lambda=args.gae_lambda, clip=args.clip, epochs=args.epochs,
                minibatch=args.minibatch, entropy_coef=args.entropy,
            )
            active_training_seconds += time.perf_counter() - t0

            if steps >= next_evaluation or steps >= args.steps:
                r = evaluate(eval_random, model, args.eval_battles)
                t = evaluate(eval_tactical, model, args.eval_battles)
                speed = steps / max(active_training_seconds, 1e-9)
                print(f"{steps:>7} decisions / {battles:>5} battles | "
                      f"random {r['win_rate']:>5.1%} | tactical {t['win_rate']:>5.1%} | "
                      f"tactical switch {t['voluntary_switches_per_battle']:.2f}/battle | "
                      f"bad-matchup switch {t['disadvantage_switch_rate']:.1%} | "
                      f"{speed:.1f} train steps/s", flush=True)
                lead_rates = {
                    f"{k}": (t["lead_results"].get(k, {}).get("wins", 0) /
                              max(1, t["lead_results"].get(k, {}).get("games", 0)))
                    for k in ("swampert vs meowscarada", "swampert vs magnezone",
                              "corviknight vs meowscarada", "corviknight vs magnezone")
                }
                print("  tactical lead wins: " + ", ".join(
                    f"{name}: {rate:.0%}" for name, rate in lead_rates.items()),
                    flush=True)
                row = dict(
                    steps=steps, train_battles=battles,
                    training_decisions_per_second=round(speed, 3),
                    vs_random_win_rate=r["win_rate"], vs_tactical_win_rate=t["win_rate"],
                    vs_random_switches_per_battle=r["voluntary_switches_per_battle"],
                    vs_tactical_switches_per_battle=t["voluntary_switches_per_battle"],
                    vs_tactical_disadvantage_switch_rate=t["disadvantage_switch_rate"],
                    vs_tactical_bad_opportunities=t["bad_opportunities"],
                    tactical_swamp_vs_meow=lead_rates["swampert vs meowscarada"],
                    tactical_swamp_vs_mag=lead_rates["swampert vs magnezone"],
                    tactical_corv_vs_meow=lead_rates["corviknight vs meowscarada"],
                    tactical_corv_vs_mag=lead_rates["corviknight vs magnezone"],
                    ppo_policy_loss=last_losses.get("policy_loss", 0),
                    ppo_value_loss=last_losses.get("value_loss", 0),
                    ppo_entropy=last_losses.get("entropy", 0),
                    ppo_approx_kl=last_losses.get("approx_kl", 0),
                )
                with csv_path.open("a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=columns).writerow(row)
                save_checkpoint(out, model, optimizer, steps, battles)
                next_evaluation = ((steps // args.eval_every) + 1) * args.eval_every
        print(f"Finished. Models: {out / 'policy_*.pt'}; "
              f"results: {csv_path}; wall time {time.perf_counter()-start:.1f}s", flush=True)
    finally:
        for e in (train_env, eval_random, eval_tactical):
            e.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("smoke", help="Benchmark random agent and inspect shape")
    s.add_argument("--battles", type=int, default=20)
    s.add_argument("--opponent", choices=["random", "tactical", "mixed"], default="tactical")
    s.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("train", help="Train fully custom masked PPO")
    s.add_argument("--steps", type=int, default=100000)
    s.add_argument("--rollout", type=int, default=512)
    s.add_argument("--eval-every", type=int, default=5000)
    s.add_argument("--eval-battles", type=int, default=200)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--gamma", type=float, default=.99)
    s.add_argument("--gae-lambda", type=float, default=.95)
    s.add_argument("--clip", type=float, default=.2)
    s.add_argument("--epochs", type=int, default=4)
    s.add_argument("--minibatch", type=int, default=64)
    s.add_argument("--entropy", type=float, default=.02)
    s.add_argument("--output", default="results_2v2_rich")
    s.add_argument("--opponent", choices=["random", "tactical", "mixed"], default="mixed")
    s.add_argument("--resume", type=str, default=None,
                   help="Continue from own training_checkpoint.pt (new battle starts)")
    s.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("evaluate", help="Evaluate an already-trained rich policy")
    s.add_argument("--weights", type=str, default="results_2v2_rich/policy.pt")
    s.add_argument("--battles", type=int, default=400)
    s.add_argument("--opponent", choices=["random", "tactical", "mixed"], default="tactical")
    s.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    seed_all(args.seed)
    if args.command == "train":
        if min(args.steps, args.rollout, args.eval_every, args.eval_battles,
               args.epochs, args.minibatch) <= 0:
            parser.error("Counts must be positive")
        train(args)
    elif args.command == "smoke":
        env = make_env(args.opponent)
        try:
            print(f"Smoke test: {N_FEATURES} observation features; "
                  f"{ACTION_COUNT} action indices", flush=True)
            stats = evaluate(env, None, args.battles, verbose=True)
            print(json.dumps(stats, indent=2), flush=True)
        finally:
            env.close()
    else:
        model = ActorCritic2()
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
        model.eval()
        env = make_env(args.opponent)
        try:
            print(f"Evaluating {args.weights} against {args.opponent}", flush=True)
            print(json.dumps(evaluate(env, model, args.battles, verbose=True), indent=2), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
