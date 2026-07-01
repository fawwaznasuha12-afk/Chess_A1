import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import chess.pgn
from tqdm import tqdm
from .board_encoder import BoardEncoder
import config

class ImitationLearning:
    """Train model with imitation learning from human datasets"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def load_pgn_dataset(self, pgn_file, max_games=None):
        """Load dataset from PGN file"""
        if max_games is None:
            max_games = config.IL_MAX_GAMES
            
        positions = []
        moves = []
        
        try:
            with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
                for game_count in range(max_games):
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                        
                    board = game.board()
                    for move in game.mainline_moves():
                        try:
                            board_tensor = BoardEncoder.encode(board)
                            positions.append(board_tensor)
                            
                            legal_moves = list(board.legal_moves)
                            if move in legal_moves:
                                move_index = legal_moves.index(move)
                                moves.append(move_index)
                                board.push(move)
                            else:
                                # Skip invalid move
                                continue
                        except Exception as e:
                            print(f"Error processing move in game {game_count}: {e}")
                            continue
            
            print(f"Loaded {len(positions)} positions from {pgn_file}")
        except FileNotFoundError:
            print(f"PGN file not found: {pgn_file}")
        except Exception as e:
            print(f"Error loading PGN dataset: {e}")
        
        return np.array(positions), np.array(moves)
    
    def train_step(self, positions, moves, batch_size=None, epochs=None):
        """Train model on positions and moves"""
        if batch_size is None:
            batch_size = config.IL_BATCH_SIZE
        if epochs is None:
            epochs = config.IL_EPOCHS
            
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=config.IL_LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        dataset_size = len(positions)
        if dataset_size == 0:
            print("No training data available!")
            return self.model
            
        num_batches = max(1, dataset_size // batch_size)
        
        for epoch in range(epochs):
            total_loss = 0
            indices = np.random.permutation(dataset_size)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                try:
                    batch_positions = torch.FloatTensor(positions[batch_indices]).to(self.device)
                    batch_moves = torch.LongTensor(moves[batch_indices]).to(self.device)
                    
                    optimizer.zero_grad()
                    policy, _ = self.model(batch_positions)
                    
                    # Ensure batch_moves are valid indices
                    if torch.max(batch_moves) >= policy.shape[1]:
                        print(f"Warning: Invalid move index {torch.max(batch_moves)} >= {policy.shape[1]}")
                        continue
                    
                    loss = criterion(policy, batch_moves)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    continue
            
            avg_loss = total_loss / max(1, num_batches)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self.model
    
    def generate_dataset_from_engine(self, num_games=100, engine_path="stockfish"):
        """Generate dataset using chess engine"""
        import chess.engine
        
        positions = []
        moves = []
        
        try:
            engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            
            for game_num in range(num_games):
                board = chess.Board()
                print(f"Generating game {game_num+1}/{num_games}")
                
                while not board.is_game_over() and len(positions) < 10000:
                    try:
                        result = engine.play(board, chess.engine.Limit(time=0.1))
                        if result.move is None:
                            break
                            
                        move = result.move
                        
                        board_tensor = BoardEncoder.encode(board)
                        positions.append(board_tensor)
                        
                        legal_moves = list(board.legal_moves)
                        if move in legal_moves:
                            move_index = legal_moves.index(move)
                            moves.append(move_index)
                            board.push(move)
                        else:
                            # Remove last position if move is invalid
                            positions.pop()
                    except Exception as e:
                        print(f"Error in engine move: {e}")
                        break
                
                if len(positions) >= 10000:
                    break
            
            engine.quit()
            print(f"Dataset generated: {len(positions)} positions")
            
        except Exception as e:
            print(f"Error generating dataset: {e}")
            print("Make sure Stockfish is installed and in PATH")
        
        return np.array(positions), np.array(moves)
