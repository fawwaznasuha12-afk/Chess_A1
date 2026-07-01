# ♟️ Chess_A1 - Chess AI with Imitation Learning & Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)]()

> "this is made by mee, for me and i dont know" — but now I know, and this is the result! 😄

---

## 📌 Table of Contents
- [About The Project](#-about-the-project)
- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Screenshots & Demo](#-screenshots--demo)
- [Future Scope](#-future-scope)
- [Team Members](#-team-members)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 📖 About The Project

**Chess_A1** is an artificial intelligence project for chess that combines three machine learning approaches:

1. **Imitation Learning (IL)** — learns from millions of master chess moves (PGN datasets)
2. **Reinforcement Learning (RL)** — learns from playing against itself (self-play)
3. **Monte Carlo Tree Search (MCTS)** — performs tree search to find the best move

I built this project as part of my learning journey in **AI and software development**. Although it's still in active development, I want to share the process and results with the open-source community.

---

## 🎯 Problem Statement

**The Problem:** Many beginners want to learn and play chess, but:
- It's hard to find a balanced opponent
- Commercial chess apps are often paid or full of ads
- Learning opportunities from chess masters are very limited

**The Solution:** An open-source chess AI that:
- Can be played offline, for free
- Provides a challenging but fair opponent for beginners
- Is built with modern AI techniques (IL + RL + MCTS)
- Serves as a learning tool for anyone interested in AI and game development

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **AI Opponent** | Play against an AI powered by MCTS + Neural Network |
| 🔄 **Self-Play Training** | The AI improves by playing millions of games against itself |
| 📚 **Imitation Learning** | Learn from grandmaster games using PGN files |
| ⚙️ **Configurable** | Easily adjust hyperparameters in `config.py` |
| 🖥️ **Terminal Interface** | Simple command-line interface for playing |
| 🐍 **Python 3.9+** | Built with modern Python practices |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.9+ |
| **Machine Learning** | PyTorch / TensorFlow (specify which one you use) |
| **Search Algorithm** | Monte Carlo Tree Search (MCTS) |
| **Data Format** | PGN (Portable Game Notation) |
| **Version Control** | Git & GitHub |
| **Environment** | Python virtual environment (venv) |

---

## 📁 Project Structure
Chess_A1/
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── config.py # Configuration & hyperparameters
├── main.py # Entry point (train / play)
├── app.py # GUI / Web interface (if applicable)
├── src/ # Source code
│ ├── init.py
│ ├── board.py # Chess board logic
│ ├── engine.py # Chess engine / game rules
│ ├── ai/ # AI modules
│ │ ├── init.py
│ │ ├── mcts.py # Monte Carlo Tree Search
│ │ ├── model.py # Neural network model
│ │ └── trainer.py # Training pipeline (IL + RL)
│ └── utils/ # Utility functions
│ └── helpers.py
├── models/ # Trained model files (.pth)
├── data/ # PGN datasets
└── tests/ # Unit tests (if any)

text

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git (optional, for cloning)

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/fawwaznasuha12-afk/Chess_A1.git
   cd Chess_A1
Create a virtual environment (recommended)

bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Verify installation

bash
python main.py --help
🎮 Usage Guide
Train the AI (Reinforcement Learning)
bash
python main.py --mode train
Play Against the AI
bash
python main.py --mode play --model models/model.pth
Imitation Learning from PGN Files
bash
python main.py --mode imitation --pgn data/sample.pgn
View All Options
bash
python main.py --help
📸 Screenshots & Demo
Coming Soon! 🖼️
Screenshots and a demo video will be added here once the interface is polished.

For now, you can run the program and see it in action on your terminal!

🔮 Future Scope
Graphical User Interface (GUI) — Build a proper GUI using Pygame or Tkinter

Web Deployment — Deploy as a web app using Streamlit or Flask

Larger Dataset — Train on a bigger PGN dataset (e.g., millions of games)

Difficulty Levels — Add easy / medium / hard modes

Opening Book — Integrate a chess opening database

Analysis Mode — Allow players to review and analyze their games

Multiplayer — Add support for two players on the same device

👥 Team Members
Name	Role	Contact
Fawwaz Nasuha	Project Lead & Developer	GitHub
This is an individual project built with passion and curiosity.

🙏 Acknowledgements
Python Chess Library — for chess logic inspiration

DeepMind AlphaZero — for the MCTS + RL concept

Open-source PGN datasets from Lichess and OpenChess

All mentors and friends who provided feedback and support

📄 License
This project is licensed under the MIT License — feel free to use, modify, and distribute it for any purpose. See the LICENSE file for details.

⭐️ Show Your Support
If you like this project, please give it a star on GitHub! ⭐
It helps me stay motivated to improve it further.

Happy Coding! 🚀
