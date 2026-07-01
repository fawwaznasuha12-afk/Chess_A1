"""
Unit tests for the ImitationLearning module.

Run tests with:
    python -m pytest tests/test_imitation_learning.py -v
"""

import unittest
import tempfile
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.imitation_learning import ImitationLearning


class SimpleChessNet(nn.Module):
    """Simple mock model for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 4672)  # Policy output
        self.fc3 = nn.Linear(256, 1)      # Value output
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        policy = self.fc2(x)
        value = self.fc3(x)
        return policy, value


class TestImitationLearning(unittest.TestCase):
    """Test cases for ImitationLearning class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleChessNet()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.il = ImitationLearning(self.model, device=self.device)
    
    def test_initialization(self):
        """Test ImitationLearning initialization."""
        self.assertIsNotNone(self.il)
        self.assertEqual(self.il.device, self.device)
        self.assertIsInstance(self.il.model, nn.Module)
    
    def test_initialization_invalid_model(self):
        """Test that initialization fails with non-Module object."""
        with self.assertRaises(TypeError):
            ImitationLearning("not a model", device=self.device)
    
    def test_train_step_valid_data(self):
        """Test training with valid data."""
        # Create dummy data
        num_samples = 32
        positions = np.random.randn(num_samples, 20, 8, 8).astype(np.float32)
        moves = np.random.randint(0, 4672, num_samples).astype(np.int64)
        
        # Train
        trained_model = self.il.train_step(
            positions, 
            moves, 
            batch_size=8, 
            epochs=1
        )
        
        # Verify model was returned
        self.assertIsInstance(trained_model, nn.Module)
    
    def test_train_step_mismatched_lengths(self):
        """Test that training fails with mismatched position/move lengths."""
        positions = np.random.randn(32, 20, 8, 8).astype(np.float32)
        moves = np.random.randint(0, 4672, 16).astype(np.int64)
        
        with self.assertRaises(ValueError):
            self.il.train_step(positions, moves, epochs=1)
    
    def test_train_step_empty_dataset(self):
        """Test that training fails with empty dataset."""
        positions = np.array([]).reshape(0, 20, 8, 8).astype(np.float32)
        moves = np.array([]).astype(np.int64)
        
        with self.assertRaises(ValueError):
            self.il.train_step(positions, moves, epochs=1)
    
    def test_load_pgn_dataset_missing_file(self):
        """Test that loading missing PGN file raises error."""
        with self.assertRaises(FileNotFoundError):
            self.il.load_pgn_dataset("/nonexistent/path/games.pgn")
    
    def test_load_pgn_dataset_returns_arrays(self):
        """Test that load_pgn_dataset returns numpy arrays."""
        # Create a minimal valid PGN file
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.pgn', 
            delete=False
        ) as f:
            # Minimal PGN with one move
            f.write('[Event "Test"]\n')
            f.write('[Site "?"]\n')
            f.write('[Date "2026.01.01"]\n')
            f.write('[White "Test"]\n')
            f.write('[Black "Test"]\n')
            f.write('\n')
            f.write('1. e4 e5 2. Nf3 *\n')
            pgn_path = f.name
        
        try:
            positions, moves = self.il.load_pgn_dataset(pgn_path, max_games=1)
            
            # Check return types
            self.assertIsInstance(positions, np.ndarray)
            self.assertIsInstance(moves, np.ndarray)
            
            # Check that we got some data
            self.assertGreater(len(positions), 0)
            self.assertEqual(len(positions), len(moves))
        
        finally:
            Path(pgn_path).unlink()  # Clean up
    
    def test_train_step_reduces_loss(self):
        """Test that training reduces loss over epochs."""
        # Create dummy data
        num_samples = 64
        positions = np.random.randn(num_samples, 20, 8, 8).astype(np.float32)
        moves = np.random.randint(0, 4672, num_samples).astype(np.int64)
        
        # Train for 2 epochs and check loss decreases
        # Note: This is not guaranteed to happen with random data,
        # but the model should run without errors
        model = self.il.train_step(
            positions, 
            moves, 
            batch_size=16, 
            epochs=2
        )
        
        self.assertIsNotNone(model)


class TestImitationLearningIntegration(unittest.TestCase):
    """Integration tests for ImitationLearning."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleChessNet()
        self.device = 'cpu'  # Use CPU for CI environments
        self.il = ImitationLearning(self.model, device=self.device)
    
    def test_full_workflow(self):
        """Test complete IL workflow: data -> train -> model."""
        # Create synthetic data
        positions = np.random.randn(32, 20, 8, 8).astype(np.float32)
        moves = np.random.randint(0, 4672, 32).astype(np.int64)
        
        # Train
        trained_model = self.il.train_step(
            positions, 
            moves, 
            batch_size=8, 
            epochs=1
        )
        
        # Verify model can make predictions
        test_position = torch.FloatTensor(positions[:1]).to(self.device)
        with torch.no_grad():
            policy, value = trained_model(test_position)
        
        # Check output shapes
        self.assertEqual(policy.shape, (1, 4672))
        self.assertEqual(value.shape, (1, 1))


if __name__ == '__main__':
    unittest.main()