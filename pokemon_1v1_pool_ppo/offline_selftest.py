"""No Showdown server needed. Test observation safety, shapes, and PPO update.

Usage: python offline_selftest.py
"""
from __future__ import annotations
from collections import OrderedDict
from enum import Enum
from types import SimpleNamespace
import numpy as np
import torch
from pool import SPECIES, static_checks
from features import N_FEATURES, battle_features, pokemon_features, POKEMON_WIDTH
from ppo import ActorCritic, ACTION_COUNT, ppo_update

class T(Enum):
    WATER=1; GROUND=2; GRASS=3; FLYING=4; STEEL=5
class C(Enum):
    PHYSICAL=1

class Mon:
    def __init__(self,name,typ=T.WATER,active=False,stats=None):
        self.species=name
        self.type_1=typ
        self.type_2=None
        self.active=active
        self.fainted=False
        self.revealed=True
        self.current_hp_fraction=.7
        self.level=100
        self.first_turn=True
        self.protect_counter=0
        self.must_recharge=False
        self.is_terastallized=False
        self.base_stats={'hp':100,'atk':110,'def':90,'spa':90,'spd':90,'spe':90}
        self.stats=stats if stats is not None else self.base_stats.copy()
        self.boosts={}
        self.status=None
        self.status_counter=0
        self.ability='torrent'
        self.item='leftovers'
        self.effects={}
        self.moves=OrderedDict()
        for m in ['earthquake','waterfall','protect','icepunch']:
            obj=SimpleNamespace(id=m,type=T.GROUND if m=='earthquake' else T.WATER,
                                category=C.PHYSICAL,base_power=100 if m!='protect' else 0,
                                accuracy=1.0,current_pp=8,max_pp=16,priority=0,
                                expected_hits=1,crit_ratio=1,heal=0,drain=0,recoil=0,
                                self_switch=False,is_protect_move=(m=='protect'),
                                is_last_used=False,secondary=None,self_boost={},boosts={},status=None)
            self.moves[m]=obj
    def damage_multiplier(self,t):
        return 2.0 if t==T.GRASS and self.type_1==T.WATER else 1.0


def test():
    static_checks()
    own=[Mon('Swampert',T.WATER,True)]
    opp=[Mon('Meowscarada',T.GRASS,True)]
    b=SimpleNamespace(team=OrderedDict((str(i),m) for i,m in enumerate(own)),
                      opponent_team=OrderedDict([('1',opp[0])]),
                      active_pokemon=own[0],opponent_active_pokemon=opp[0],
                      available_switches=[],
                      available_moves=list(own[0].moves.values()),
                      turn=1,trapped=False,can_tera=False,used_tera=False,
                      opponent_used_tera=False,weather={},fields={},
                      side_conditions={},opponent_side_conditions={})
    a=battle_features(b)
    assert a.shape==(N_FEATURES,) and np.isfinite(a).all() and np.abs(a).max()<=1
    assert pokemon_features(None,owned=False)==[0.0]*POKEMON_WIDTH
    # Changing unrevealed foe raw stats must not change observation.
    before=a.copy()
    opp[0].stats={'hp':20,'atk':10000,'def':1,'spa':10000,'spd':1,'spe':10000}
    assert np.array_equal(before,battle_features(b)), 'Hidden opponent stats leaked!'
    # Unrevealed opponent moves must be legal (partial information).
    opp[0].moves=OrderedDict() # Partial information must be legal.
    a=battle_features(b)
    assert a.shape==(N_FEATURES,)
    net=ActorCritic()
    opt=torch.optim.Adam(net.parameters(),lr=3e-4)
    mask=np.array([False]*6 + [True]*4 + [False]*16)  # 1v1: no switches
    sample={k:[] for k in ('obs','mask','action','logp','value','reward','done')}
    for i in range(32):
        inp=torch.from_numpy(a).float().unsqueeze(0)
        m=torch.from_numpy(mask).unsqueeze(0)
        with torch.no_grad():
            d,v=net(inp,m)
            action=d.sample()
        sample['obs'].append(a)
        sample['mask'].append(mask)
        sample['action'].append(action.item())
        sample['logp'].append(d.log_prob(action).item())
        sample['value'].append(v.item())
        sample['reward'].append(1.0 if i%4==0 else 0.0)
        sample['done'].append(i%8==7)
    result=ppo_update(net,opt,sample,0.0)
    assert np.isfinite(list(result.values())).all()
    print('OFFLINE CHECKS PASS:',N_FEATURES,'observation features;',
          sum(p.numel() for p in net.parameters()),'PPO parameters')

if __name__=='__main__':test()
