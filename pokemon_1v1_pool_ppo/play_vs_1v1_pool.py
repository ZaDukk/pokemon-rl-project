r"""Challenge the saved 1v1-pool PPO model from a local browser.

Example from repository root:
  .\.venv\Scripts\python.exe -u .\pokemon_1v1_pool_ppo\play_vs_1v1_pool.py --human HumanPlayer1 --weights runs\1v1_pool6\policy.pt --mon garchomp

Use the '[Gen 9] RL 1v1' custom format and a single Pokemon (any of the 12 sets in pool.py).
"""
from __future__ import annotations
import argparse
import random
import asyncio
from pathlib import Path
import numpy as np
import torch
from poke_env import AccountConfiguration
from poke_env.environment import SinglesEnv
from poke_env.player import Player
from poke_env.player.battle_order import DefaultBattleOrder
from features import N_FEATURES, battle_features, seed_all
from pool import DEFAULT_POOL, pack_team, sample_mon
from ppo import ACTION_COUNT, ActorCritic, choose_action
from prototype_1v1_pool import FORMAT


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
    # A fresh Pokemon is drawn for EACH battle. No fixed seed by default, so
    # every run differs; pass --seed N to make the sequence reproducible.
    rng=random.Random(a.seed)
    if a.seed is not None:
        seed_all(a.seed)
    model=ActorCritic()
    model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True))
    model.eval()
    pick=lambda: sample_mon(DEFAULT_POOL,rng,name=a.mon)
    bot=TrainedPPOPlayer(
        model=model, account_configuration=AccountConfiguration(a.bot_name,None),
        battle_format=FORMAT,team=pack_team(pick()),open_timeout=None,log_level=30,
    )
    for i in range(a.battles):
        mon=pick()
        bot.update_team(pack_team(mon))
        print(f'\n=== Battle {i+1}/{a.battles}: bot Pokemon is {mon} ===',flush=True)
        if a.random_human:
            print(f'    Suggested team for you: {rng.choice(DEFAULT_POOL)} '
                  '(pick your saved team for it when accepting)',flush=True)
        if a.accept:
            print('Waiting for a challenge from',a.human,flush=True)
            await bot.accept_challenges(a.human,1)
        else:
            print('Challenging',a.human,flush=True)
            await bot.send_challenges(a.human,n_challenges=1)
    print(f'\nBot won {bot.n_won_battles}/{bot.n_finished_battles} completed games.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--human',required=True)
    p.add_argument('--weights',default='runs/1v1_pool/policy.pt')
    p.add_argument('--bot-name',default='PoolPPOBot')
    p.add_argument('--mon',default=None,
                   help='Fix the bot Pokemon, e.g. garchomp (default: new random one each battle)')
    p.add_argument('--seed',type=int,default=None,help='Make the random Pokemon sequence repeatable')
    p.add_argument('--random-human',action='store_true',
                   help='Also suggest a random Pokemon for YOU each battle')
    p.add_argument('--battles',type=int,default=1)
    p.add_argument('--accept',action='store_true')
    a=p.parse_args()
    if a.battles<1:p.error('--battles must be >=1')
    if a.human.lower()==a.bot_name.lower():p.error('Human and bot usernames must differ')
    asyncio.run(run(a))

if __name__=='__main__':main()
