"""Restricted Gen 9 3v3 Singles with 12-Pokemon pool + HAND-WRITTEN PPO.

Training curriculum: random/max-power/type-aware scripted opponents, followed by
historical self-play against frozen previous PPO checkpoints (not simultaneous
self-play). Works with poke-env==0.16.1 and local Showdown custom format.

  python prototype_3v3.py check-pool
  python prototype_3v3.py smoke --battles 12
  python prototype_3v3.py train --steps 10000 --eval-every 5000
  python prototype_3v3.py train --steps 80000 --resume runs/3v3/training_checkpoint.pt
  python prototype_3v3.py evaluate --weights runs/3v3/policy.pt --opponent typebot --battles 200
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv, SingleAgentWrapper
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer

from pool import SETS, SPECIES, pack_team, sample_team, static_checks
from features import N_FEATURES, TEAM_SIZE, bad_matchup, battle_features, seed_all, species
from opponents import MixedOpponent, TypeAwarePlayer
from ppo import ACTION_COUNT, ActorCritic3, choose_action, as_tensors, ppo_update

FORMAT = 'gen9rl3v3singles'
MAX_TURNS = 120


class RichThreeVsThreeEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(low=-1.0,high=1.0,shape=(N_FEATURES,),dtype=np.float32)
            for agent in self.possible_agents
        }

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return battle_features(battle)

    def calc_reward(self, battle: AbstractBattle) -> float:
        # Same reward WEIGHTS as previous 2v2, using correct 3-mon team size.
        return self.reward_computing_helper(
            battle, number_of_pokemons=TEAM_SIZE,
            hp_value=1.0, fainted_value=1.0, victory_value=5.0
        )


def make_env(opponent: str) -> SingleAgentWrapper:
    base = RichThreeVsThreeEnv(
        battle_format=FORMAT,
        team=pack_team(tuple(SPECIES[:3])),
        choose_on_teampreview=False,
        log_level=40,
        open_timeout=10.0,
        challenge_timeout=45.0,
    )
    base.agent2.update_team(pack_team(tuple(SPECIES[3:6])))
    opponent_kwargs = dict(battle_format=FORMAT,start_listening=False)
    opponents = {
        'random': RandomPlayer,
        'maxpower': MaxBasePowerPlayer,
        'heuristic': SimpleHeuristicsPlayer,
        'typebot': TypeAwarePlayer,
        'curriculum': MixedOpponent,
    }
    cls = opponents.get(opponent)
    if cls is None: raise ValueError(f'Unknown opponent {opponent}')
    return SingleAgentWrapper(base,cls(**opponent_kwargs))


def reset_teams(env, rng=None, ours=None, theirs=None):
    ours = sample_team(rng,names=ours)
    theirs = sample_team(rng,names=theirs)
    # Team order is significant: first is lead, and 0..5 switch action mapping
    # follows the insertion order in battle.team.
    env.env.agent1.update_team(pack_team(ours))
    env.env.agent2.update_team(pack_team(theirs))
    obs, info = env.reset()
    return obs, info, ours, theirs


def step_safely(env, action):
    """If >120 turns, use a REAL Showdown forfeit to end the battle.

    Never reset an unfinished env, which can hang the poke-env event loop.
    Fainting/reward from forfeit are added to the preceding agent transition.
    """
    obs,reward,terminated,truncated,info=env.step(np.int64(action))
    done=bool(terminated or truncated)
    forced_forfeit=0
    if not done and env.env.battle1.turn >= MAX_TURNS:
        obs2,r2,t2,tc2,info2=env.step(np.int64(-1))
        if not (t2 or tc2):
            raise RuntimeError('Forfeit at the turn cap failed: battle still live. Do not reset.')
        obs, reward, done, info=obs2, float(reward)+float(r2), True, info2
        forced_forfeit=1
    return obs,float(reward),done,forced_forfeit


def describe_action(battle, a):
    if a < 6:
        mons=list(battle.team.values())
        return f'SWITCH {species(mons[a]) if a<len(mons) else "?"}'
    if a<10 and battle.active_pokemon:
        moves=list(battle.active_pokemon.moves.values())
        return moves[a-6].id if a-6<len(moves) else 'MOVE?'
    return f'action {a}'


def evaluate(env, model, games, *, seed=2027, verbose=False, trace=0):
    """Evaluate on random 3-of-12 teams, no training or weight updates."""
    rng=random.Random(seed)
    total=Counter()
    matchups=defaultdict(lambda:[0,0])
    start=time.perf_counter()
    for i in range(games):
        obs,_,ours,theirs=reset_teams(env,rng=rng)
        b=env.env.battle1
        matchup=f'{ours[0]} vs {theirs[0]}'
        switches=0
        decisions=0
        while True:
            b=env.env.battle1
            action,_,_=choose_action(model,obs,deterministic=True)
            mask=np.asarray(obs['action_mask'],dtype=bool)
            if not mask[action]:
                raise RuntimeError(f'Masked policy selected illegal action {action}')
            if i<trace:
                print(f'  [game {i+1} turn {b.turn}] {species(b.active_pokemon)} vs '
                      f'{species(b.opponent_active_pokemon)}: {describe_action(b,action)}',flush=True)
            attack_legal=bool(mask[6:10].any())
            switch_legal=bool(mask[:6].any())
            if action<6:
                if attack_legal:
                    total['voluntary_switches']+=1
                    switches+=1
                else:
                    total['forced_switches']+=1
            if bad_matchup(b) and switch_legal and attack_legal:
                total['bad_opportunities']+=1
                if action<6:total['bad_switches']+=1
            obs,_,done,forced=step_safely(env,action)
            total['cap_forfeits']+=forced
            decisions+=1
            if done: break
        b=env.env.battle1
        win=int(bool(b.won))
        total['wins']+=win
        total['ties']+=int(not b.won and not b.lost)
        total['decisions']+=decisions
        total['turns']+=b.turn
        matchups[matchup][0]+=win
        matchups[matchup][1]+=1
        if verbose and (i+1)%max(1,games//10)==0:
            print(f'  evaluated {i+1}/{games}',flush=True)
    elapsed=time.perf_counter()-start
    return {
        'win_rate':total['wins']/games,
        'ties':total['ties'],
        'battles':games,
        'voluntary_switches_per_battle':total['voluntary_switches']/games,
        'forced_switches_per_battle':total['forced_switches']/games,
        'bad_matchup_switch_rate':total['bad_switches']/max(1,total['bad_opportunities']),
        'bad_opportunities':total['bad_opportunities'],
        'decisions_per_battle':total['decisions']/games,
        'turns_per_battle':total['turns']/games,
        'cap_forfeits':total['cap_forfeits'],
        'battles_per_second':games/max(elapsed,1e-9),
        'elapsed_seconds':round(elapsed,2),
        'lead_results':{k:{'wins':v[0],'games':v[1]} for k,v in matchups.items()},
    }


def empty_batch():
    return {x:[] for x in ('obs','mask','action','logp','value','reward','done')}


def save_checkpoint(out,model,optimizer,steps,battles,opponent,*,resume_config):
    out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out/f'policy_{steps:08d}.pt')
    torch.save(model.state_dict(),out/'policy.pt')
    snapshots=[(s,m.state_dict()) for s,m in opponent.snapshots]
    torch.save({
        'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
        'steps':steps,'battles':battles,'snapshots':snapshots,
        'py_random':random.getstate(),'numpy_random':np.random.get_state(),
        'torch_random':torch.get_rng_state(),
        'n_features':N_FEATURES,'pool':SPECIES,'format':FORMAT,
        'resume_config':resume_config,
    },out/'training_checkpoint.pt')


def train(args):
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    model=ActorCritic3()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    steps=battles=0
    ckpt=None
    if args.resume:
        ckpt=torch.load(args.resume,map_location='cpu',weights_only=False)
        if (ckpt['n_features'],tuple(ckpt['pool']))!=(N_FEATURES,SPECIES):
            raise ValueError('Cannot resume: observation or Pokemon pool changed.')
        original=ckpt.get('resume_config',{})
        current=dict(opponent=args.opponent,selfplay_start=args.selfplay_start,
                     historical_prob=args.historical_prob,rollout=args.rollout,
                     lr=args.lr)
        if original!=current:
            raise ValueError(f'Resume config differs! Saved {original}; now {current}. '
                             'Keep hyperparameters and training mode the same.')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        steps,battles=int(ckpt['steps']),int(ckpt['battles'])
        print(f'Resuming at {steps} decisions; a fresh battle starts (simulator states not serialized).',flush=True)
    resume_config=dict(opponent=args.opponent,selfplay_start=args.selfplay_start,
                       historical_prob=args.historical_prob,rollout=args.rollout,
                       lr=args.lr)
    meta=dict(poke_env='0.16.1',format=FORMAT,pool=list(SPECIES),team_size=3,
              state_features=N_FEATURES,action_count=ACTION_COUNT,
              reward=dict(hp=1,faint=1,win=5),
              algorithm='our PPO with clipped loss, GAE, legal action masking',
              opponent_sampling='scripted bots + historical frozen PPO snapshots',
              args=vars(args))
    (out/'run_config.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    train_env=make_env('curriculum')
    opponents=['random','maxpower','typebot']
    eval_envs={name:make_env(name) for name in opponents}
    mixer=train_env.opponent
    assert isinstance(mixer,MixedOpponent)
    if ckpt is not None:
        for old_steps,state in ckpt.get('snapshots',[]):
            m=ActorCritic3();m.load_state_dict(state);m.eval()
            mixer.snapshots.append((old_steps,m))
        random.setstate(ckpt['py_random'])
        np.random.set_state(ckpt['numpy_random'])
        torch.set_rng_state(ckpt['torch_random'])
    results=out/'results.csv'
    fields=['steps','train_battles','train_decisions_per_second',
            'vs_random','vs_maxpower','vs_typebot',
            'typebot_switches_per_battle','typebot_bad_matchup_switch_rate',
            'typebot_turns_per_battle','typebot_cap_forfeits',
            'snapshots','train_random','train_maxpower','train_typebot','train_historical',
            'ppo_policy_loss','ppo_value_loss','ppo_entropy','ppo_approx_kl']
    if not results.exists() or steps==0:
        with results.open('w',newline='',encoding='utf-8') as f:
            csv.DictWriter(f,fieldnames=fields).writeheader()
    training_secs=0.0
    started=time.perf_counter()
    next_eval=((steps//args.eval_every)+1)*args.eval_every
    last_losses={}

    def reset_train():
        mixer.choose_for_next_battle(steps,strategy=args.opponent,
                                      start=args.selfplay_start,
                                      historical_prob=args.historical_prob)
        obs,_,_,_=reset_teams(train_env)
        return obs
    try:
        print(f'3v3: {len(SPECIES)}-Pokemon pool; random teams of 3 each; '
              f'{N_FEATURES} observation features; {sum(p.numel() for p in model.parameters()):,} '
              f'PPO parameters.',flush=True)
        if steps==0:
            print('Untrained baselines (random legal action agent):',flush=True)
            for label, e in eval_envs.items():
                baseline=evaluate(e,None,args.eval_battles,seed=args.eval_seed)
                print(f'  vs {label}: {baseline["win_rate"]:.1%} wins',flush=True)
        obs=reset_train()
        while steps<args.steps:
            batch=empty_batch()
            t0=time.perf_counter()
            for _ in range(min(args.rollout,args.steps-steps)):
                action,logp,value=choose_action(model,obs)
                batch['obs'].append(obs['observation'].copy())
                batch['mask'].append(obs['action_mask'].copy())
                batch['action'].append(action)
                batch['logp'].append(logp)
                batch['value'].append(value)
                obs,reward,done,forfeited=step_safely(train_env,action)
                batch['reward'].append(float(reward))
                batch['done'].append(bool(done))
                steps+=1
                if done:
                    battles+=1
                    obs=reset_train()
            if batch['done'][-1]:
                bootstrap=0.0
            else:
                with torch.no_grad():
                    x,mask=as_tensors(obs)
                    _,v=model(x,mask)
                    bootstrap=float(v.item())
            last_losses=ppo_update(model,optimizer,batch,bootstrap,
                                   gamma=args.gamma,gae_lambda=args.gae_lambda,
                                   clip=args.clip,epochs=args.epochs,
                                   minibatch=args.minibatch,entropy_coef=args.entropy)
            training_secs+=time.perf_counter()-t0
            if steps>=next_eval or steps>=args.steps:
                metrics={name:evaluate(e,model,args.eval_battles,seed=args.eval_seed)
                         for name,e in eval_envs.items()}
                scores=' | '.join(f'{name} {metrics[name]["win_rate"]:.1%}' for name in opponents)
                print(f'{steps:>8} decisions / {battles:>5} training battles | {scores} | '
                      f'typebot switches {metrics["typebot"]["voluntary_switches_per_battle"]:.2f}/battle '
                      f'| {steps/max(training_secs,1e-9):.1f} steps/s '
                      f'| snapshots {len(mixer.snapshots)}',flush=True)
                row=dict(steps=steps,train_battles=battles,
                         train_decisions_per_second=round(steps/max(training_secs,1e-9),2),
                         vs_random=metrics['random']['win_rate'],
                         vs_maxpower=metrics['maxpower']['win_rate'],
                         vs_typebot=metrics['typebot']['win_rate'],
                         typebot_switches_per_battle=metrics['typebot']['voluntary_switches_per_battle'],
                         typebot_bad_matchup_switch_rate=metrics['typebot']['bad_matchup_switch_rate'],
                         typebot_turns_per_battle=metrics['typebot']['turns_per_battle'],
                         typebot_cap_forfeits=metrics['typebot']['cap_forfeits'],
                         snapshots=len(mixer.snapshots),
                         **{f'train_{k}':mixer.mode_counts[k] for k in mixer.mode_counts},
                         **{f'ppo_{k}':last_losses.get(k,0.0) for k in
                            ('policy_loss','value_loss','entropy','approx_kl')})
                with results.open('a',newline='',encoding='utf-8') as f:
                    csv.DictWriter(f,fieldnames=fields).writerow(row)
                # The next training games can sample a frozen historical opponent.
                mixer.add_snapshot(model,steps)
                save_checkpoint(out,model,optimizer,steps,battles,mixer,resume_config=resume_config)
                next_eval=((steps//args.eval_every)+1)*args.eval_every
        print(f'Finished. Checkpoints: {out/"policy_*.pt"}; logs: {results}; '
              f'elapsed {time.perf_counter()-started:.1f}s.',flush=True)
    finally:
        train_env.close()
        for e in eval_envs.values():e.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('info',help='Offline roster and feature-info check; no Showdown required.')
    check=sub.add_parser('check-pool',help='Test all 12 sets against local Showdown.')
    check.add_argument('--battles',type=int,default=4)
    smoke=sub.add_parser('smoke',help='Short live simulator benchmark vs random player')
    smoke.add_argument('--battles',type=int,default=12)
    train_parser=sub.add_parser('train',help='Train hand-written PPO with scripted bots and historical self-play')
    train_parser.add_argument('--steps',type=int,default=10000)
    train_parser.add_argument('--rollout',type=int,default=256)
    train_parser.add_argument('--eval-every',type=int,default=5000)
    train_parser.add_argument('--eval-battles',type=int,default=60)
    train_parser.add_argument('--eval-seed',type=int,default=2027)
    train_parser.add_argument('--output',default='runs/3v3')
    train_parser.add_argument('--resume',default=None)
    train_parser.add_argument('--opponent',choices=['curriculum','random','maxpower','typebot','historical'],default='curriculum')
    train_parser.add_argument('--selfplay-start',type=int,default=15000)
    train_parser.add_argument('--historical-prob',type=float,default=.35)
    train_parser.add_argument('--lr',type=float,default=3e-4)
    train_parser.add_argument('--gamma',type=float,default=.99)
    train_parser.add_argument('--gae-lambda',type=float,default=.95)
    train_parser.add_argument('--clip',type=float,default=.2)
    train_parser.add_argument('--epochs',type=int,default=4)
    train_parser.add_argument('--minibatch',type=int,default=64)
    train_parser.add_argument('--entropy',type=float,default=.01)
    train_parser.add_argument('--seed',type=int,default=42)
    evaluation=sub.add_parser('evaluate',help='Evaluate weights against never-trained fixed baselines')
    evaluation.add_argument('--weights',required=True)
    evaluation.add_argument('--opponent',choices=['random','maxpower','typebot','heuristic'],default='typebot')
    evaluation.add_argument('--battles',type=int,default=200)
    evaluation.add_argument('--seed',type=int,default=2027)
    evaluation.add_argument('--trace-battles',type=int,default=0)
    args=parser.parse_args()
    if args.command=='info':
        print(json.dumps(dict(**static_checks(),N_FEATURES=N_FEATURES,
                              N_ACTIONS=ACTION_COUNT,team_size=TEAM_SIZE,
                              pool=list(SPECIES)),indent=2))
        return
    if args.command=='train':
        if args.steps<=0 or args.rollout<=0 or args.eval_every<=0 or args.eval_battles<=0:
            parser.error('steps, rollout, eval-every and eval-battles must be positive')
        if not 0 <= args.historical_prob <=1:
            parser.error('--historical-prob must be 0 to 1')
        seed_all(args.seed)
        train(args)
        return
    seed_all(getattr(args,'seed',42))
    if args.command=='check-pool':
        env=make_env('random')
        try:
            for i in range(max(4,args.battles)):
                ours=tuple(SPECIES[j%len(SPECIES)] for j in range(3*i,3*i+3))
                theirs=tuple(SPECIES[j%len(SPECIES)] for j in range(3*i+3,3*i+6))
                obs,_,_,_=reset_teams(env,ours=ours,theirs=theirs)
                done=False
                while not done:
                    a,_,_=choose_action(None,obs)
                    obs,_,done,_=step_safely(env,a)
                print(f'  valid group {i+1}: {ours} vs {theirs}',flush=True)
            print('All 12 Pokemon tested with Showdown team validation and complete battles.',flush=True)
        finally:env.close()
        return
    if args.command=='smoke':
        env=make_env('random')
        try:
            print(json.dumps(evaluate(env,None,args.battles,verbose=True),indent=2))
        finally:env.close()
        return
    if args.command=='evaluate':
        model=ActorCritic3()
        model.load_state_dict(torch.load(args.weights,map_location='cpu',weights_only=True))
        model.eval()
        env=make_env(args.opponent)
        try:
            print(json.dumps(evaluate(env,model,args.battles,seed=args.seed,
                                      verbose=True,trace=args.trace_battles),indent=2))
        finally:env.close()


if __name__=='__main__':main()
