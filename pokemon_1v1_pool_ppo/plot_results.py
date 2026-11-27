"""Plot learning curves from a run; matplotlib is an optional dependency."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--csv',default='runs/1v1_pool/results.csv')
    a.add_argument('--out',default='runs/1v1_pool/learning_curves.png')
    args=a.parse_args()
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader(open(args.csv,encoding='utf-8')))
    if not rows:raise ValueError('No checkpoint evaluation rows yet')
    xs=[int(r['steps']) for r in rows]
    for label,key in [('RandomPlayer','vs_random'),('MaxBasePowerPlayer','vs_maxpower'),('TypeAwarePlayer','vs_typebot')]:
        plt.plot(xs,[float(r[key])*100 for r in rows],marker='o',label=label)
    plt.xlabel('Training decisions')
    plt.ylabel('Win rate (%)')
    plt.ylim(0,100)
    plt.grid(alpha=.25)
    plt.legend()
    plt.title('1v1 PPO – one Pokémon each from a fixed pool')
    plt.tight_layout()
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(path,dpi=160)
    print(f'Saved {path}')

if __name__=='__main__':main()
