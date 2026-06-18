#!/usr/bin/env python
"""
Main program for Chess AI combining Imitation Learning,
Reinforcement Learning, and MCTS
"""

import torch
import numpy as np
import argparse
import os
import time

from src.board_encoder import BoardEncoder
from src.neural_network import ChessNet
from src.mcts import MCTS
from src.self_play import SelfPlay
from src.imitation_learning import ImitationLearning
from src.trainer import Trainer
from src.game import Game, ChessGUI
from src.utils import Utils
from src.db_handler import DatabaseHandler
import config

def main():
    parser = argparse.ArgumentParser(description='Chess AI with IL + RL + MCTS')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'save_db', 'play', 'evaluate', 'imitation', 'db_list', 'db_load', 'db_delete'],
                        help='Operation mode')
    
    # Parameter Training
    parser.add_argument('--games', type=int, default=100,
                        help='Number of self-play games (default: 100)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--simulations', type=int, default=200,
                        help='Number of MCTS simulations (default: 200)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of training iterations (default: 10)')
    
    # Model
    parser.add_argument('--model', type=str, default='models/model.pth',
                        help='Model path (default: models/model.pth)')
    parser.add_argument('--model_name', type=str, default='chess_ai_model',
                        help='Model name in database (default: chess_ai_model)')
    
    # Database
    parser.add_argument('--db_load', type=int, default=None,
                        help='Model ID to load from database')
    
    # Imitation Learning
    parser.add_argument('--pgn', type=str, default=None,
                        help='PGN file for Imitation Learning')
    
    args = parser.parse_args()
    
    device = Utils.get_device()
    print(f"Using device: {device}")
    
    # Initialize database
    db = DatabaseHandler(
        host=config.DB_HOST,
        port=config.DB_PORT,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        database=config.DB_NAME
    )
    
    # ============================================
    # MODE: db_list - List all models in database
    # ============================================
    if args.mode == 'db_list':
        if db.connect():
            db.create_tables()
            db.list_models()
            db.close()
        return
    
    # ============================================
    # MODE: db_delete - Delete model from database
    # ============================================
    if args.mode == 'db_delete':
        if args.db_load:
            if db.connect():
                db.delete_model(args.db_load)
                db.close()
        else:
            print("❌ Use --db_load [ID] to delete a model")
        return
    
    # ============================================
    # MODE: db_load - Load model from database
    # ============================================
    if args.mode == 'db_load':
        if args.db_load:
            if db.connect():
                model, metadata = db.load_model(args.db_load)
                db.close()
                if model:
                    print(f"✅ Model loaded from database! ID: {args.db_load}")
                    print(f"📊 Metadata: {metadata}")
                    # Save locally
                    Utils.save_model(model, args.model)
                    print(f"✅ Model saved to: {args.model}")
                else:
                    print(f"❌ Model with ID {args.db_load} not found")
            else:
                print("❌ Failed to connect to database")
        else:
            print("❌ Use --db_load [ID] to load a model")
        return
    
    # ============================================
    # MODEL INITIALIZATION
    # ============================================
    model = ChessNet()
    
    # Load model from file if exists
    if os.path.exists(args.model) and args.mode != 'db_load':
        model, _ = Utils.load_model(model, args.model)
    
    # ============================================
    # MODE: imitation - Imitation Learning
    # ============================================
    if args.mode == 'imitation':
        print(f"\n{'='*50}")
        print("IMITATION LEARNING")
        print(f"{'='*50}")
        
        il = ImitationLearning(model, device)
        
        if args.pgn:
            print(f"📂 Loading PGN: {args.pgn}")
            positions, moves = il.load_pgn_dataset(args.pgn)
        else:
            print("🤖 Generating dataset from Stockfish...")
            positions, moves = il.generate_dataset_from_engine(num_games=50)
        
        if len(positions) > 0:
            print(f"📊 Dataset size: {len(positions)} positions")
            model = il.train_step(positions, moves, epochs=args.epochs)
            Utils.save_model(model, args.model)
            print(f"\n✅ Model saved to: {args.model}")
        else:
            print("❌ No training data available")
        return
    
    # ============================================
    # MODE: train - Training + save locally
    # ============================================
    if args.mode == 'train':
        print(f"\n{'='*50}")
        print("TRAINING MODE")
        print(f"{'='*50}")
        print(f"Games per iteration: {args.games}")
        print(f"Epochs: {args.epochs}")
        print(f"MCTS Simulations: {args.simulations}")
        print(f"Iterations: {args.iterations}")
        print(f"{'='*50}")
        
        trainer = Trainer(model, device)
        start_time = time.time()
        
        for iteration in range(args.iterations):
            print(f"\n--- Iteration {iteration+1}/{args.iterations} ---")
            
            selfplay = SelfPlay(model, simulations=args.simulations)
            positions, probs, values = selfplay.generate_training_data(num_games=args.games)
            
            Utils.create_training_data_file(positions, probs, values, 
                                          f"data/training_data_iter_{iteration}.npz")
            
            model = trainer.train_loop(positions, probs, values, epochs=args.epochs)
            
            Utils.save_model(model, f"models/model_iter_{iteration}.pth")
            
            Utils.log_message(f"Iteration {iteration+1} completed with {len(positions)} positions")
        
        # Save final model
        Utils.save_model(model, args.model)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"✅ Iterations: {args.iterations}")
        print(f"✅ Time: {elapsed/60:.1f} minutes")
        print(f"✅ Model saved at: {args.model}")
        print(f"{'='*50}")
        return
    
    # ============================================
    # MODE: save_db - Save model to database
    # ============================================
    if args.mode == 'save_db':
        print(f"\n{'='*50}")
        print("SAVE TO DATABASE")
        print(f"{'='*50}")
        
        if not os.path.exists(args.model):
            print(f"❌ Model file not found: {args.model}")
            print(f"💡 Run training first: --mode train")
            return
        
        print(f"📤 Loading model from: {args.model}")
        model = ChessNet()
        model, _ = Utils.load_model(model, args.model)
        
        print(f"📤 Connecting to database...")
        if db.connect():
            db.create_tables()
            model_id = db.save_model(
                model, 
                args.model_name,
                iteration=args.iterations,
                epoch=args.epochs
            )
            if model_id:
                print(f"\n✅ MODEL SAVED TO DATABASE!")
                print(f"✅ ID: {model_id}")
                print(f"✅ Name: {args.model_name}")
                print(f"✅ Iterations: {args.iterations}")
                print(f"✅ Epochs: {args.epochs}")
            else:
                print("❌ Failed to save model to database")
            db.close()
        else:
            print("❌ Failed to connect to database")
        return
    
    # ============================================
    # MODE: play - Play against AI
    # ============================================
    if args.mode == 'play':
        print(f"\n{'='*50}")
        print("PLAY MODE")
        print(f"{'='*50}")
        print(f"Model: {args.model}")
        print(f"Simulations: {args.simulations}")
        print(f"{'='*50}")
        
        if not os.path.exists(args.model):
            print(f"❌ Model file not found: {args.model}")
            print(f"💡 Run training first: --mode train")
            return
        
        gui = ChessGUI(model)
        gui.play_game()
        return
    
    # ============================================
    # MODE: evaluate - Evaluate model
    # ============================================
    if args.mode == 'evaluate':
        print(f"\n{'='*50}")
        print("EVALUATE MODE")
        print(f"{'='*50}")
        
        if not os.path.exists(args.model):
            print(f"❌ Model file not found: {args.model}")
            print(f"💡 Run training first: --mode train")
            return
        
        model2 = ChessNet()
        if os.path.exists(args.model):
            model2, _ = Utils.load_model(model2, args.model)
        
        trainer = Trainer(model, device)
        game = Game()
        wins, losses, draws = trainer.evaluate(model2, game.board, num_games=10)
        
        print(f"\n{'='*50}")
        print("EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Wins: {wins}")
        print(f"Losses: {losses}")
        print(f"Draws: {draws}")
        win_rate = wins / (wins + losses + draws) * 100 if (wins + losses + draws) > 0 else 0
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"{'='*50}")
        return

if __name__ == '__main__':
    main()