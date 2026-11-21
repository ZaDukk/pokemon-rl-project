r"""Challenge the saved 3v3 PPO model from a local browser.

Example from starter project root:
  .\.venv\Scripts\python.exe -u .\pokemon_3v3_pool_ppo\play_vs_3v3.py --human HumanPlayer1 --weights runs\3v3\policy.pt

Use custom '[Gen 9] RL 3v3 Singles' Showdown format and a 3-Pokemon team.
"""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path
import numpy as np
import torch
from poke_env import AccountConfiguration
from poke_env.environment import SinglesEnv
from poke_env.player import Player
from poke_env.player.battle_order import DefaultBattleOrder
from features import N_FEATURES, battle_features, seed_all
from pool import SETS, SPECIES, pack_team, sample_team
from ppo import ACTION_COUNT, ActorCritic3, choose_action
from prototype_3v3 import FORMAT


class TrainedPPOPlayer(Player):
    def __init__(self, *, model, **kwargs):
        super().__init__(**kwargs)
        self.model=model

    def choose_move(self,battle):
        if battle.wait:return DefaultBattleOrder()
        mask=np.asarray(SinglesEnv.get_action_mask(battle),dtype=np.int8)
        if mask.shape!=(ACTION_COUNT,) or not mask.any():
            print('No valid masked actions; using random fallback',flush=True)
            return self.choose_random_move(battle)
        obs={'observation':battle_features(battle),'action_mask':mask}
        a,_,_=choose_action(self.model,obs,deterministic=True)
        order=SinglesEnv.action_to_order(np.int64(a),battle)
        print(f'  Turn {battle.turn}: action {a}: {order.message}',flush=True)
        return order


async def run(a):
    seed_all(a.seed)
    model=ActorCritic3()
    model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True))
    model.eval()
    names=tuple(x.strip().lower().replace('-','') for x in a.team.split(',')) if a.team else sample_team()
    team=pack_team(sample_team(names=names))
    print('Bot team:',names,flush=True)
    bot=TrainedPPOPlayer(
        model=model, account_configuration=AccountConfiguration(a.bot_name,None),
        battle_format=FORMAT,team=team,open_timeout=None,log_level=30,
    )
    if a.accept:
        print('Waiting for challenges from',a.human,flush=True)
        await bot.accept_challenges(a.human,a.battles)
    else:
        print('Challenging',a.human,flush=True)
        await bot.send_challenges(a.human,n_challenges=a.battles)
    print(f'Bot won {bot.n_won_battles}/{bot.n_finished_battles} completed games.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--human',required=True)
    p.add_argument('--weights',default='runs/3v3/policy.pt')
    p.add_argument('--bot-name',default='PoolPPOBot')
    p.add_argument('--team',default='swampert,corviknight,garchomp',
                   help='Three comma-separated names from the 12-species pool')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--battles',type=int,default=1)
    p.add_argument('--accept',action='store_true')
    a=p.parse_args()
    if a.battles<1:p.error('--battles must be >=1')
    if a.human.lower()==a.bot_name.lower():p.error('Human and bot usernames must differ')
    asyncio.run(run(a))

if __name__=='__main__':main()
