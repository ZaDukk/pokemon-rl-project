"""Scripted opponents and frozen historical PPO opponents for single-agent RL.

No opponent looks at a hidden team slot. Historical PPO gets exactly the same
battle_features() as the learner, but from its own player perspective.
"""
from __future__ import annotations
import copy
import random
from collections import deque
import numpy as np
import torch
from poke_env.environment import SinglesEnv
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from features import battle_features, matchup_type_multiplier
from ppo import ActorCritic


def known_damage_score(move, owner, foe) -> float:
    """Simple visible-information attack heuristic, NOT exact damage calculation."""
    try:
        bp = float(move.base_power or 0)
        if bp <= 0:
            return 0.0
        mult = float(foe.damage_multiplier(move.type)) if foe is not None else 1.0
        stab = 1.5 if move.type in (owner.type_1, owner.type_2) else 1.0
        acc = float(move.accuracy) if move.accuracy is not None else 1.0
        return bp * mult * stab * max(0.0, min(acc, 1.0))
    except (TypeError, ValueError, AttributeError, KeyError):
        return 0.0


def typebot_order(player, battle):
    """Untrained rule-based reference opponent: type-aware attacking + cautious switches."""
    actor = battle.active_pokemon
    foe = battle.opponent_active_pokemon
    if actor is None or actor.fainted:
        if battle.available_switches:
            return player.create_order(max(battle.available_switches, key=lambda m:
                max((known_damage_score(move, m, foe) for move in m.moves.values()),default=0.0)))
        return player.choose_random_move(battle)
    attacks = [m for m in battle.available_moves if float(m.base_power or 0) > 0]
    best = max(attacks, key=lambda m: known_damage_score(m, actor, foe), default=None)
    score = known_damage_score(best, actor, foe) if best else 0.0
    try:
        danger = max((float(actor.damage_multiplier(t)) for t in (foe.type_1, foe.type_2) if t is not None), default=1.0) if foe else 1.0
    except (ValueError, KeyError):
        danger = 1.0
    if battle.available_switches and not battle.trapped:
        options=[]
        for mon in battle.available_switches:
            try:
                their_stab = max((float(mon.damage_multiplier(t)) for t in (foe.type_1,foe.type_2) if t), default=1.0) if foe else 1.0
            except (ValueError, KeyError):
                their_stab = 1.0
            best_attack = max((known_damage_score(m, mon, foe) for m in mon.moves.values()),default=0.0)
            factor = (1.0 + 0.3 * max(0.0, 2.0 - their_stab)) * max(.2, float(mon.current_hp_fraction))
            options.append((best_attack * factor, their_stab, mon))
        if options:
            other_score, other_danger, mon = max(options, key=lambda x:x[0])
            # Avoid endless back-and-forth: switch only for a substantial improvement.
            turns = battle.turn
            last_switch = getattr(player, '_last_switch_by_tag', {}).get(battle.battle_tag, -99)
            if (turns - last_switch >= 3 and
                (danger >= 2.0 or score <= 50.0) and
                other_score > max(score * 1.4, 60.0) and other_danger < danger):
                player._last_switch_by_tag[battle.battle_tag] = turns
                return player.create_order(mon)
    if best is not None:
        return player.create_order(best)
    if battle.available_moves:
        return player.create_order(battle.available_moves[0])
    return player.choose_random_move(battle)


class TypeAwarePlayer(RandomPlayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_switch_by_tag = {}

    def choose_move(self, battle):
        return typebot_order(self, battle)


class MixedOpponent(RandomPlayer):
    """Used by learner's SingleAgentWrapper. Select mode ONCE per battle.

    `historical` is frozen PPO (a past checkpoint); it is not simultaneous
    two-agent PPO, which would have different training dynamics.
    """
    def __init__(self, *, max_history=6, **kwargs):
        super().__init__(**kwargs)
        self.max_history = max_history
        self.snapshots: deque[tuple[int, ActorCritic]] = deque(maxlen=max_history)
        self.mode = 'random'
        self.current_snapshot = None
        self._last_switch_by_tag = {}
        self.mode_counts = {'random':0,'maxpower':0,'typebot':0,'historical':0}

    def add_snapshot(self, model: ActorCritic, steps: int):
        frozen = ActorCritic()
        frozen.load_state_dict(copy.deepcopy(model.state_dict()))
        frozen.eval()
        for p in frozen.parameters(): p.requires_grad_(False)
        self.snapshots.append((steps,frozen))

    def choose_for_next_battle(self, steps:int, *, strategy:str, start:int, historical_prob:float):
        if strategy == 'curriculum':
            prob = historical_prob if steps >= start and self.snapshots else 0.0
            # Remaining mixture retains a strong fixed-opponent anchor.
            non = 1-prob
            modes=['random','maxpower','typebot','historical']
            weights=[non*.20,non*.30,non*.50,prob]
            self.mode = random.choices(modes,weights=weights,k=1)[0]
        elif strategy == 'historical':
            self.mode = 'historical' if self.snapshots else 'typebot'
        else:
            self.mode = strategy
        self.current_snapshot = random.choice(list(self.snapshots))[1] if self.mode=='historical' else None
        self.mode_counts[self.mode] += 1

    @torch.no_grad()
    def choose_move(self,battle):
        if self.mode == 'random':
            return self.choose_random_move(battle)
        if self.mode == 'maxpower':
            if battle.available_moves:
                return self.create_order(max(battle.available_moves,key=lambda m: float(m.base_power or 0)))
            return self.choose_random_move(battle)
        if self.mode == 'typebot':
            return typebot_order(self,battle)
        if self.mode == 'historical':
            if self.current_snapshot is None:
                return typebot_order(self,battle)
            state = torch.tensor(battle_features(battle),dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(SinglesEnv.get_action_mask(battle),dtype=torch.bool).unsqueeze(0)
            dist,_ = self.current_snapshot(state,mask)
            action = np.int64(dist.probs.argmax(dim=-1).item())
            return SinglesEnv.action_to_order(action,battle)
        raise ValueError(f'Unexpected opponent mode {self.mode}')
