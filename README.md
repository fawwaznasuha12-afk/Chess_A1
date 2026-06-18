Chess AI

This repository contains a Chess AI combining imitation learning, reinforcement learning, and MCTS (Monte Carlo Tree Search).

Requirements:
- Python 3.9+
- See requirements.txt for full dependencies.

Quick start:
- Train: python main.py --mode train
- Play: python main.py --mode play --model models/model.pth
- Imitation learning from PGN: python main.py --mode imitation --pgn data/sample.pgn

Configuration: edit config.py for hyperparameters and paths.

License: MIT
