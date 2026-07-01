import chess
import numpy as np
from tqdm import tqdm
from .board_encoder import BoardEncoder
from .mcts import MCTS

class SelfPlay:
    """Manage the self-play process for reinforcement learning"""
    
    def __init__(self, model, simulations=200, temperature=1.0):
        self.model = model
        self.simulations = simulations
        self.temperature = temperature
        self.mcts = MCTS(model, simulations, temperature=temperature)
        
    def play_game(self, temperature_start=1.0, temperature_end=0.5, max_moves=300):
        """Play a single self-play game"""
        board = chess.Board()
        game_data = []
        
        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            temperature = temperature_start - (temperature_start - temperature_end) * (move_count / 80)
            temperature = max(temperature, temperature_end)
            self.mcts.temperature = temperature
            
            result = self.mcts.search(board)
            if result is None:
                break
                
            best_move, action_probs = result
            
            try:
                board_tensor = BoardEncoder.encode(board)
                legal_moves = list(board.legal_moves)
                
                if not legal_moves:
                    break
                
                probs = np.zeros(len(legal_moves))
                for action, prob in action_probs:
                    try:
                        action_index = legal_moves.index(action)
                        probs[action_index] = prob
                    except ValueError:
                        continue
                
                game_data.append({
                    'position': board_tensor,
                    'probs': probs,
                    'move': best_move,
                    'legal_moves': legal_moves
                })
                
                board.push(best_move)
                move_count += 1
            except Exception as e:
                print(f"Error in game play: {e}")
                break
        
        # Determine winner and assign values
        result = board.result()
        if result == "1-0":
            winner = 1.0
        elif result == "0-1":
            winner = -1.0
        else:
            winner = 0.0
        
        for i, data in enumerate(game_data):
            value = winner if i % 2 == 0 else -winner
            data['value'] = value
        
        return game_data
    
    def generate_training_data(self, num_games=100):
        """Generate training data from self-play games"""
        all_data = []
        
        for game_num in tqdm(range(num_games), desc="Playing self-play games"):
            try:
                game_data = self.play_game()
                all_data.extend(game_data)
            except Exception as e:
                print(f"Error in game {game_num}: {e}")
                continue
        
        if len(all_data) == 0:
            print("No training data generated!")
            return np.array([]), np.array([]), np.array([])
        
        positions = []
        probs = []
        values = []
        
        for d in all_data:
            positions.append(d['position'])
            probs.append(d['probs'])
            values.append(d['value'])
        
        positions = np.array(positions, dtype=np.float32)
        probs = np.array(probs, dtype=object)
        values = np.array(values, dtype=np.float32)
        
        print(f"Generated {len(positions)} training positions")
        return positions, probs, values
