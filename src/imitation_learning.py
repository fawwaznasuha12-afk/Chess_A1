import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import chess.pgn
from tqdm import tqdm
from .board_encoder import BoardEncoder

class ImitationLearning:
    """Train model with imitation learning from human datasets"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def load_pgn_dataset(self, pgn_file, max_games=1000):
        positions = []
        moves = []
        
        with open(pgn_file, 'r') as f:
            for _ in range(max_games):
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                    
                board = game.board()
                for move in game.mainline_moves():
                    board_tensor = BoardEncoder.encode(board)
                    positions.append(board_tensor)
                    
                    move_index = list(board.legal_moves).index(move)
                    moves.append(move_index)
                    
                    board.push(move)
        
        return np.array(positions), np.array(moves)
    
    def train_step(self, positions, moves, batch_size=64, epochs=5):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        dataset_size = len(positions)
        num_batches = dataset_size // batch_size
        
        for epoch in range(epochs):
            total_loss = 0
            indices = np.random.permutation(dataset_size)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                
                batch_positions = torch.FloatTensor(positions[batch_indices]).to(self.device)
                batch_moves = torch.LongTensor(moves[batch_indices]).to(self.device)
                
                optimizer.zero_grad()
                policy, _ = self.model(batch_positions)
                loss = criterion(policy, batch_moves)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}")
        
        return self.model
    
    def generate_dataset_from_engine(self, num_games=100, engine_path="stockfish"):
        import chess.engine
        
        positions = []
        moves = []
        
        try:
            engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            
            for game_num in range(num_games):
                board = chess.Board()
                print(f"Generating game {game_num+1}/{num_games}")
                
                while not board.is_game_over():
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    move = result.move
                    
                    board_tensor = BoardEncoder.encode(board)
                    positions.append(board_tensor)
                    
                    move_index = list(board.legal_moves).index(move)
                    moves.append(move_index)
                    
                    board.push(move)
                
                if len(positions) > 10000:
                    break
            
            engine.quit()
            print(f"Dataset generated: {len(positions)} positions")
            
        except Exception as e:
            print(f"Error generating dataset: {e}")
            print("Make sure Stockfish is installed and in PATH")
        
        return np.array(positions), np.array(moves)
