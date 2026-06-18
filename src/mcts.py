import numpy as np
import chess
import math
from collections import defaultdict

class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, state, parent=None, action=None, prior=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        
        self.visit_count = 0
        self.total_value = 0
        self.children = {}
        self.is_expanded = False
        
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count
    
    def ucb_score(self, c_puct=1.5):
        if self.visit_count == 0:
            return float('inf')
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value() + exploration

class MCTS:
    """Monte Carlo Tree Search for Chess"""
    
    def __init__(self, model, simulations=200, c_puct=1.5, temperature=1.0):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        
    def search(self, board):
        root = MCTSNode(board.copy())
        
        for _ in range(self.simulations):
            node = root
            path = [node]
            
            while node.is_expanded and node.children:
                action, node = max(node.children.items(), key=lambda x: x[1].ucb_score(self.c_puct))
                path.append(node)
            
            if not node.state.is_game_over():
                from .board_encoder import BoardEncoder
                board_tensor = BoardEncoder.encode(node.state)
                
                policy, value = self.model.predict(board_tensor)
                node.is_expanded = True
                
                moves = list(node.state.legal_moves)
                for i, move in enumerate(moves):
                    new_state = node.state.copy()
                    new_state.push(move)
                    if i < len(policy):
                        prior = policy[i]
                        node.children[move] = MCTSNode(new_state, node, move, prior)
            else:
                result = node.state.result()
                if result == "1-0":
                    value = 1.0
                elif result == "0-1":
                    value = -1.0
                else:
                    value = 0.0
            
            for node in reversed(path):
                node.visit_count += 1
                node.total_value += value
                value = -value
        
        if not root.children:
            return None
            
        actions = list(root.children.keys())
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        
        if self.temperature == 0:
            best_action = actions[np.argmax(visit_counts)]
            probs = np.zeros(len(actions))
            probs[np.argmax(visit_counts)] = 1.0
        else:
            probs = np.exp(visit_counts / self.temperature)
            probs = probs / np.sum(probs)
            best_action = np.random.choice(actions, p=probs)
        
        return best_action, list(zip(actions, probs))
