import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class Trainer:
    """Trainer for the chess model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    def train_step(self, positions, probs, values, batch_size=256):
        self.model.train()
        
        dataset_size = len(positions)
        if dataset_size == 0:
            print("No training data!")
            return 0
            
        num_batches = max(1, dataset_size // batch_size)
        total_loss = 0
        
        indices = np.random.permutation(dataset_size)
        for batch_idx in tqdm(range(num_batches), desc="Training batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            batch_positions = torch.FloatTensor(np.array([positions[i] for i in batch_indices])).to(self.device)
            
            max_probs_len = max([len(probs[i]) for i in batch_indices])
            batch_probs = np.zeros((len(batch_indices), max_probs_len))
            for i, idx in enumerate(batch_indices):
                batch_probs[i, :len(probs[idx])] = probs[idx]
            batch_probs = torch.FloatTensor(batch_probs).to(self.device)
            
            batch_values = torch.FloatTensor([values[i] for i in batch_indices]).unsqueeze(1).to(self.device)
            
            self.optimizer.zero_grad()
            policy, value = self.model(batch_positions)
            
            policy = policy[:, :max_probs_len]
            
            policy_loss = -torch.mean(torch.sum(batch_probs * policy, dim=1))
            value_loss = nn.MSELoss()(value, batch_values)
            loss = policy_loss + value_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train_loop(self, positions, probs, values, epochs=10, batch_size=256):
        if len(positions) == 0:
            print("No training data, skipping training!")
            return self.model
            
        for epoch in range(epochs):
            loss = self.train_step(positions, probs, values, batch_size)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
            
            if (epoch + 1) % 5 == 0:
                torch.save(self.model.state_dict(), f"models/model_epoch_{epoch+1}.pth")
        
        return self.model
    
    def evaluate(self, model2, board, num_games=10):
        wins = 0
        losses = 0
        draws = 0
        
        from .mcts import MCTS
        from .game import Game
        
        for _ in range(num_games):
            game = Game()
            game.play_game_with_mcts(self.model, model2)
            
            if game.winner == 1:
                wins += 1
            elif game.winner == -1:
                losses += 1
            else:
                draws += 1
        
        print(f"Results: {wins} wins, {losses} losses, {draws} draws")
        return wins, losses, draws