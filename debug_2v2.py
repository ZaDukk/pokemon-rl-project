r"""Diagnose a stuck poke-env 2v2 test. Place next to prototype_2v2.py.

Run: .\.venv\Scripts\python.exe -u debug_2v2.py

Prints a progress line before and after each reset/step. If a call blocks,
faulthandler prints Python thread stack traces every 20 seconds.
"""

import faulthandler
import logging
import random
import sys
import time

import numpy as np

from prototype_2v2 import make_env, reset_random_leads, species


def say(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main():
    random.seed(42)
    np.random.seed(42)
    faulthandler.enable()
    faulthandler.dump_traceback_later(20, repeat=True, file=sys.stderr)
    say('Creating 2v2 environment...')
    env = make_env('tactical')
    try:
        # During diagnostics show significant poke-env errors.
        env.env.agent1.logger.setLevel(logging.WARNING)
        env.env.agent2.logger.setLevel(logging.WARNING)
        say('Environment created. Calling reset_random_leads()...')
        obs, _ = reset_random_leads(env)
        say('Reset returned: the battle has started.')
        done = False
        decision = 0
        while not done:
            battle = env.env.battle1
            action_mask = np.asarray(obs['action_mask'])
            legal = np.flatnonzero(action_mask)
            say(f'Turn {battle.turn}, decision {decision}: '
                f'{species(battle.active_pokemon)} vs {species(battle.opponent_active_pokemon)}; '
                f'legal actions={legal.tolist()}')
            if not len(legal):
                raise RuntimeError('Empty legal action mask')

            # Prevent the diagnostic itself from running into endless switching.
            if battle.turn >= 30 or decision >= 60:
                action = -1  # Valid poke-env 'forfeit battle' action.
                say('Safety limit reached. Forfeiting the diagnostic battle.')
            else:
                action = int(np.random.choice(legal))

            say(f'Calling env.step(action={action})...')
            obs, reward, terminated, truncated, _ = env.step(np.int64(action))
            decision += 1
            done = bool(terminated or truncated)
            say(f'env.step returned: reward={reward:.3f}, '
                f'terminated={terminated}, truncated={truncated}')
        say(f'Completed one battle! Won={env.env.battle1.won}, '
            f'Lost={env.env.battle1.lost}, decisions={decision}')
    finally:
        faulthandler.cancel_dump_traceback_later()
        say('Closing environment...')
        env.close()
        say('Closed.')


if __name__ == '__main__':
    main()
