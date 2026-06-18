import numpy as np
import chess

class BoardEncoder:
    
    @staticmethod
    def encode(board):
        """
        Convert board position to a numpy tensor
        Shape: (20, 8, 8)
        """
        tensor = np.zeros((20, 8, 8), dtype=np.float32)
        
        piece_map = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                color_offset = 0 if piece.color == chess.WHITE else 6
                channel = color_offset + piece_map[piece.piece_type]
                tensor[channel, row, col] = 1.0
        
        castling = board.castling_rights
        tensor[12, :, :] = 1.0 if castling & chess.BB_A1 else 0.0
        tensor[13, :, :] = 1.0 if castling & chess.BB_H1 else 0.0
        tensor[14, :, :] = 1.0 if castling & chess.BB_A8 else 0.0
        tensor[15, :, :] = 1.0 if castling & chess.BB_H8 else 0.0
        
        ep = board.ep_square
        if ep:
            row, col = divmod(ep, 8)
            tensor[16, row, col] = 1.0
        
        tensor[17, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
        
        return tensor
    
    @staticmethod
    def decode_action(action_index, board):
        moves = list(board.legal_moves)
        if action_index < len(moves):
            return moves[action_index]
        return None
    
    @staticmethod
    def get_action_size(board):
        return len(list(board.legal_moves))
