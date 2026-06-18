import chess
import pygame
import os
from .mcts import MCTS

class Game:
    def __init__(self):
        self.board = chess.Board()
        self.history = []
        self.winner = None

    def play_game_with_mcts(self, model1, model2=None):
        self.board = chess.Board()
        self.history = []

        mcts1 = MCTS(model1, simulations=100)
        mcts2 = MCTS(model2, simulations=100) if model2 else mcts1

        move_count = 0
        while not self.board.is_game_over() and move_count < 200:
            if self.board.turn == chess.WHITE:
                result = mcts1.search(self.board)
            else:
                result = mcts2.search(self.board)

            if result is None:
                break

            best_move, _ = result
            self.board.push(best_move)
            self.history.append(best_move)
            move_count += 1

        if self.board.is_game_over():
            if self.board.result() == "1-0":
                self.winner = 1
            elif self.board.result() == "0-1":
                self.winner = -1
            else:
                self.winner = 0
        else:
            self.winner = 0

        return self.winner


class ChessGUI:
    def __init__(self, model):
        self.model = model
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.square_size = 80
        self.offset = 50

        pygame.init()
        self.screen = pygame.display.set_mode((8 * self.square_size + 2 * self.offset,
                                               8 * self.square_size + 2 * self.offset))
        pygame.display.set_caption("Chess AI")
        self.clock = pygame.time.Clock()

        # Use Segoe UI Symbol font (supports Unicode chess symbols)
        try:
            self.font = pygame.font.SysFont("segoeuisymbol", self.square_size - 10)
        except:
            self.font = pygame.font.Font(None, self.square_size - 10)

        self.piece_map = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        }

    def draw_board(self):
        colors = [(240, 217, 181), (181, 136, 99)]

        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                rect = pygame.Rect(
                    self.offset + col * self.square_size,
                    self.offset + row * self.square_size,
                    self.square_size,
                    self.square_size
                )
                pygame.draw.rect(self.screen, color, rect)

                if self.selected_square:
                    sel_row, sel_col = divmod(self.selected_square, 8)
                    if row == sel_row and col == sel_col:
                        pygame.draw.rect(self.screen, (255, 255, 0), rect, 5)

                for move in self.legal_moves:
                    if move.to_square == row * 8 + col:
                        center = rect.center
                        pygame.draw.circle(self.screen, (0, 0, 0), center, 15)

    def draw_pieces(self):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                symbol = piece.symbol()

                text = self.font.render(self.piece_map[symbol], True, (0, 0, 0))
                text_rect = text.get_rect(center=(
                    self.offset + col * self.square_size + self.square_size // 2,
                    self.offset + row * self.square_size + self.square_size // 2
                ))
                self.screen.blit(text, text_rect)

    def get_square_from_mouse(self, mouse_pos):
        x, y = mouse_pos
        col = (x - self.offset) // self.square_size
        row = (y - self.offset) // self.square_size

        if 0 <= row < 8 and 0 <= col < 8:
            return row * 8 + col
        return None

    def play_game(self):
        running = True
        player_color = chess.WHITE
        ai_color = chess.BLACK

        mcts = MCTS(self.model, simulations=200)

        while running:
            if self.board.turn == ai_color and not self.board.is_game_over():
                result = mcts.search(self.board)
                if result:
                    best_move, _ = result
                    self.board.push(best_move)
                    self.selected_square = None
                    self.legal_moves = []

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and self.board.turn == player_color:
                    mouse_pos = pygame.mouse.get_pos()
                    square = self.get_square_from_mouse(mouse_pos)

                    if square is not None:
                        if self.selected_square is None:
                            piece = self.board.piece_at(square)
                            if piece and piece.color == player_color:
                                self.selected_square = square
                                self.legal_moves = [m for m in self.board.legal_moves
                                                   if m.from_square == square]
                        else:
                            move = chess.Move(self.selected_square, square)

                            piece = self.board.piece_at(self.selected_square)
                            if piece and piece.piece_type == chess.PAWN:
                                if (piece.color == chess.WHITE and square // 8 == 7) or \
                                   (piece.color == chess.BLACK and square // 8 == 0):
                                    move = chess.Move(self.selected_square, square,
                                                    promotion=chess.QUEEN)

                            if move in self.board.legal_moves:
                                self.board.push(move)
                                self.selected_square = None
                                self.legal_moves = []
                            else:
                                self.selected_square = None
                                self.legal_moves = []

            self.screen.fill((0, 0, 0))
            self.draw_board()
            self.draw_pieces()

            if self.board.is_game_over():
                result = self.board.result()
                font = pygame.font.Font(None, 36)
                text = font.render(f"Game Over: {result}", True, (255, 255, 255))
                self.screen.blit(text, (self.offset, 10))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()