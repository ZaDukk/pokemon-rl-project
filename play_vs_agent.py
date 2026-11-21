r"""Play against the trained Pokémon 1v1 PPO agent on a LOCAL Showdown server.

Put this file alongside prototype.py and results/policy.pt.
Run from the starter directory:

    .\.venv\Scripts\python.exe play_vs_agent.py --human YourShowdownName

Open http://localhost:8000 and connect to the SAME local server before running.
The bot will send you a challenge in the custom '[Gen 9] RL 1v1' format.

This does not train the network or modify your saved weights.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import torch
from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv
from poke_env.player import Player
from poke_env.player.battle_order import DefaultBattleOrder

from prototype import (
    ActorCritic,
    BATTLE_FORMAT,
    N_FEATURES,
    TEAM,
    boost,
    choose_action,
    hp,
    seed_all,
)


def observation_from_battle(battle: AbstractBattle) -> np.ndarray:
    """Exactly the 7 features and order used for PPO training."""
    ours, theirs = battle.active_pokemon, battle.opponent_active_pokemon
    return np.array(
        [
            hp(ours), hp(theirs),
            boost(ours, "atk"), boost(theirs, "atk"),
            boost(ours, "def"), boost(theirs, "def"),
            min(float(battle.turn) / 40.0, 1.0),
        ],
        dtype=np.float32,
    )


class TrainedConkeldurr(Player):
    def __init__(self, model: ActorCritic, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def choose_move(self, battle: AbstractBattle):
        if battle.wait:
            return DefaultBattleOrder()

        observation = observation_from_battle(battle)
        mask = np.asarray(SinglesEnv.get_action_mask(battle), dtype=np.int8)
        if observation.shape != (N_FEATURES,) or mask.shape != (26,):
            raise RuntimeError(
                f"Unexpected feature/action shape: {observation.shape}, {mask.shape}"
            )
        if not np.any(mask):
            return self.choose_random_move(battle)

        action, _, _ = choose_action(
            self.model,
            {"observation": observation, "action_mask": mask},
            deterministic=True,
        )
        order = SinglesEnv.action_to_order(np.int64(action), battle)
        print(f"Turn {battle.turn}: action {action} ({order.message})", flush=True)
        return order


async def run(args):
    seed_all(args.seed)
    path = Path(args.weights)
    if not path.is_file():
        raise FileNotFoundError(f"Saved policy not found: {path}")

    model = ActorCritic(n_actions=26)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()

    bot = TrainedConkeldurr(
        model=model,
        account_configuration=AccountConfiguration(args.bot_name, None),
        battle_format=BATTLE_FORMAT,
        team=TEAM,
        open_timeout=None,
        log_level=30,
    )
    print(f"Loaded {path} as bot '{args.bot_name}'")
    print("Make sure you are connected to the LOCAL Showdown server (localhost:8000).")
    print(f"Custom battle format: {BATTLE_FORMAT}; team: one Conkeldurr")
    if args.accept:
        print(f"Waiting for {args.human} to challenge {args.bot_name} ...", flush=True)
        await bot.accept_challenges(args.human, args.battles)
    else:
        print(f"Sending {args.battles} challenge(s) to '{args.human}' ...", flush=True)
        await bot.send_challenges(args.human, n_challenges=args.battles)

    print("Finished!")
    print(f"Bot record: {bot.n_won_battles} wins / {bot.n_finished_battles} battles")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", required=True, help="Your username on the LOCAL Showdown server")
    parser.add_argument("--bot-name", default="RLConkBot", help="Bot's local Showdown username")
    parser.add_argument("--weights", default="results/policy.pt")
    parser.add_argument("--battles", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--accept", action="store_true",
        help="Wait for your challenge instead of sending a challenge to you",
    )
    args = parser.parse_args()
    if args.battles < 1:
        parser.error("--battles must be >= 1")
    if args.human.lower() == args.bot_name.lower():
        parser.error("Human and bot usernames must be different")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
