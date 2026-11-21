r"""Play the trained rich-observation 2v2 PPO model against a human locally.

Requirements:
    - Your existing local Pokemon Showdown server must be running.
    - The custom gen9rl2v2singles format must already be installed.
    - Put this script beside the RICH version of prototype_2v2.py.

In a browser open http://localhost:8000, choose a username, and prepare a
2-Pokemon team for the custom '[Gen 9] RL 2v2 Singles' singles format.

PowerShell example (from the project root):
    .\.venv\Scripts\python.exe -u play_vs_rich_2v2.py --human HumanPlayer1

This DOES NOT train the agent or change the saved weights.
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

from prototype_2v2 import (
    ACTION_COUNT,
    FORMAT,
    MY_TEAMS,
    N_FEATURES,
    ActorCritic2,
    battle_features,
    choose_action,
    seed_all,
)


class RichPPOBot(Player):
    """Adapts the exact rich training observation + action mask to Player."""

    def __init__(self, model: ActorCritic2, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def choose_move(self, battle: AbstractBattle):
        if battle.wait:
            return DefaultBattleOrder()

        mask = np.asarray(SinglesEnv.get_action_mask(battle), dtype=np.int8)
        if mask.shape != (ACTION_COUNT,):
            raise RuntimeError(f"Unexpected action mask {mask.shape}; expected {(ACTION_COUNT,)}")
        if not mask.any():
            # Exceptional state: return a legal fallback instead of calling PPO
            # with an empty mask.
            print(f"Battle turn {battle.turn}: no legal PPO action; using fallback", flush=True)
            return self.choose_random_move(battle)

        features = battle_features(battle)
        if features.shape != (N_FEATURES,):
            raise RuntimeError(f"Unexpected observation shape: {features.shape}")

        action, _, _ = choose_action(
            self.model,
            {"observation": features, "action_mask": mask},
            deterministic=True,
        )
        if not mask[action]:
            raise AssertionError(f"PPO selected illegal action {action}")

        # IMPORTANT: poke-env 0.16.1 calls .item() on this argument in its
        # action_to_order implementation. np.int64 works; Python int does not.
        order = SinglesEnv.action_to_order(np.int64(action), battle)
        print(
            f"Turn {battle.turn:>3}: PPO action {action:>2} -> {order.message}",
            flush=True,
        )
        return order


async def run(args):
    seed_all(args.seed)
    path = Path(args.weights)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Check that you're running from the starter project folder and "
            "that the 17,800-step checkpoint exists."
        )

    model = ActorCritic2()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()

    bot = RichPPOBot(
        model=model,
        account_configuration=AccountConfiguration(args.bot_name, None),
        battle_format=FORMAT,
        team=MY_TEAMS[0 if args.lead == "swampert" else 1],
        open_timeout=None,
        log_level=30,
    )

    print(f"Loaded checkpoint: {path}", flush=True)
    print(f"Bot username: {args.bot_name}", flush=True)
    print(f"Format: {FORMAT} (2v2 singles); lead: {args.lead}", flush=True)
    print("Browser: http://localhost:8000 (the SAME local server)", flush=True)

    if args.accept:
        print(f"Waiting for a challenge from '{args.human}' ...", flush=True)
        await bot.accept_challenges(args.human, args.battles)
    else:
        print(f"Sending {args.battles} challenge(s) to '{args.human}' ...", flush=True)
        await bot.send_challenges(args.human, n_challenges=args.battles)

    print(
        f"Done: bot won {bot.n_won_battles} of "
        f"{bot.n_finished_battles} completed battles.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", required=True, help="Your username in the LOCAL Showdown browser")
    parser.add_argument("--bot-name", default="RichPPOBot", help="Name displayed for the PPO bot")
    parser.add_argument(
        "--weights", default=r"results_2v2_rich_test\policy_0017800.pt",
        help="Path to rich 2v2 ActorCritic2 checkpoint",
    )
    parser.add_argument("--lead", choices=("swampert", "corviknight"), default="swampert")
    parser.add_argument("--battles", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--accept", action="store_true", help="Wait for you to challenge the bot instead"
    )
    args = parser.parse_args()
    if args.battles < 1:
        parser.error("--battles must be at least 1")
    if args.human.strip().lower() == args.bot_name.strip().lower():
        parser.error("Choose different human and bot usernames")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
