import pygame
import sys
import time
import os
import copy
from rwkv_go_infer_model import infer_from_sequence

def board_to_text_representation(board, next_player):
    """
    Converts the Pygame board object into a string representation
    for the inference model.
    """
    player_map = {0: '#', PLAYER_BLACK: 'B', PLAYER_WHITE: 'W'}
    board_str = ""
    for y in range(board.size):
        row_str = "".join([player_map[board.grid[y][x]] for x in range(board.size)])
        board_str += row_str + "\n"
    
    player_color = "Black" if next_player == PLAYER_BLACK else "White"
    return board_str.strip() + f"\n{player_color}"


# --- PYGAME FRONTEND START --- (Most remains the same)
BOARD_SIZE = 19
GRID_WIDTH = 60 # Reduced for a better fit on some screens
MARGIN = 50
INFO_PANEL_HEIGHT = 100
WINDOW_WIDTH = 2 * MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH
WINDOW_HEIGHT = 2 * MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH + INFO_PANEL_HEIGHT

BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
BOARD_COLOR = (218, 165, 32)
LINE_COLOR = (50, 50, 50)
TEXT_COLOR = (10, 10, 10)
BUTTON_COLOR = (100, 100, 200)
BUTTON_HOVER_COLOR = (150, 150, 250)
BUTTON_TEXT_COLOR = WHITE_COLOR

PLAYER_BLACK = 1
PLAYER_WHITE = 2

# Game Modes
MODE_MENU = 0
MODE_PLAYER_IS_BLACK = 1
MODE_PLAYER_IS_WHITE = 2

COORDS_X = "ABCDEFGHIJKLMNOPQRS"
# COORDS_Y = "abcdefghijklmnopqrs"
COORDS_Y = "srqponmlkjihgfedcba"


def to_notation(point):
    if point is None: return None
    if point == 'PASS': return 'PASS'
    x, y = point
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE): return ""
    return COORDS_X[x] + COORDS_Y[y]


def from_notation(notation):
    """Converts Go notation (e.g., 'D4', 'Qh') to (x, y) coordinates."""
    if not isinstance(notation, str) or len(notation) < 2: return None
    notation = notation.strip()
    
    if len(notation) != 2: return None

    col_char = notation[0].upper()
    row_char = notation[1].lower()

    if col_char not in COORDS_X or row_char not in COORDS_Y: return None

    x = COORDS_X.index(col_char)
    y = COORDS_Y.index(row_char)
    
    return x, y

class Board:
    def __init__(self, size):
        self.size = size
        self.reset()

    def reset(self):
        self.grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.history = []
        self.ko_point = None

    def pass_turn(self):
        self.history.append('PASS')

    def is_valid_move(self, x, y, player):
        if not (0 <= x < self.size and 0 <= y < self.size): return False
        if self.grid[y][x] != 0: return False
        if (x, y) == self.ko_point: return False
        
        # Create a temporary board to test for suicide
        temp_grid = [row[:] for row in self.grid]
        temp_grid[y][x] = player
        
        # Check if the move captures any opponent stones
        opponent = 3 - player
        captures_made = False
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and temp_grid[ny][nx] == opponent:
                group, liberties = self._find_group_on_board(nx, ny, temp_grid)
                if not liberties:
                    captures_made = True
                    break
        
        if captures_made:
            return True # Move is valid because it captures

        # If no captures, check if the move itself has liberties (not suicide)
        _, final_liberties = self._find_group_on_board(x, y, temp_grid)
        if not final_liberties:
            return False # Suicidal move

        return True

    def place_stone(self, x, y, player):
        if not self.is_valid_move(x, y, player): return []

        self.grid[y][x] = player
        self.history.append((x, y))
        
        captured_stones = self._capture_stones(x, y, player)
        
        # Simplified Ko logic
        self.ko_point = None
        if len(captured_stones) == 1:
            # Check if the move resulted in a Ko situation
            group, liberties = self._find_group(x, y)
            if len(group) == 1 and not liberties:
                self.ko_point = captured_stones[0]

        return captured_stones
        
    def _capture_stones(self, x, y, player):
        captured_stones = []
        opponent = 3 - player
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[ny][nx] == opponent:
                group, liberties = self._find_group(nx, ny)
                if not liberties:
                    for stone_x, stone_y in group:
                        if (stone_x, stone_y) not in captured_stones:
                            self.grid[stone_y][stone_x] = 0
                            captured_stones.append((stone_x, stone_y))
        return captured_stones

    def _find_group_on_board(self, x, y, board_grid):
        """Finds the group of connected stones and their liberties on a given grid."""
        player = board_grid[y][x]
        if player == 0: return [], []
        q, visited, group, liberties = [(x, y)], set([(x, y)]), [], set()
        while q:
            cx, cy = q.pop(0)
            group.append((cx, cy))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if board_grid[ny][nx] == 0: liberties.add((nx, ny))
                    elif board_grid[ny][nx] == player and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group, list(liberties)

    def _find_group(self, x, y):
        """Helper to find a group on the current main board."""
        return self._find_group_on_board(x, y, self.grid)

class GameUI:
    def __init__(self, board):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Go Game - Human vs AI")
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        self.board = board
        self.game_mode = MODE_MENU
        self.current_player = PLAYER_BLACK
        self.status_text = "Welcome! Select a game mode."
        self.last_move = None
        self.pass_count = 0
        self.game_over = False
        self.ai_player = None
        self.human_player = None
        self.ai_is_thinking = False

    def draw_board(self):
        self.screen.fill(BOARD_COLOR)
        for i in range(BOARD_SIZE):
            # Vertical and horizontal lines
            start_pos_h = (MARGIN + i * GRID_WIDTH, MARGIN)
            end_pos_h = (MARGIN + i * GRID_WIDTH, MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH)
            pygame.draw.line(self.screen, LINE_COLOR, start_pos_h, end_pos_h)
            start_pos_v = (MARGIN, MARGIN + i * GRID_WIDTH)
            end_pos_v = (MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH, MARGIN + i * GRID_WIDTH)
            pygame.draw.line(self.screen, LINE_COLOR, start_pos_v, end_pos_v)
        
        # Star points
        star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        for p in star_points:
            pos = (MARGIN + p[0] * GRID_WIDTH, MARGIN + p[1] * GRID_WIDTH)
            pygame.draw.circle(self.screen, LINE_COLOR, pos, 5)
            
        # Coordinates
        for i in range(BOARD_SIZE):
            text = self.small_font.render(COORDS_X[i], True, TEXT_COLOR)
            self.screen.blit(text, (MARGIN + i * GRID_WIDTH - text.get_width() // 2, MARGIN - 35))
            text = self.small_font.render(str(BOARD_SIZE - i), True, TEXT_COLOR) # Numbers from 19 down to 1
            self.screen.blit(text, (MARGIN - 35, MARGIN + i * GRID_WIDTH - text.get_height() // 2))

    def draw_stones(self):
        for y in range(self.board.size):
            for x in range(self.board.size):
                player = self.board.grid[y][x]
                if player != 0:
                    color = BLACK_COLOR if player == PLAYER_BLACK else WHITE_COLOR
                    pos = (MARGIN + x * GRID_WIDTH, MARGIN + y * GRID_WIDTH)
                    pygame.draw.circle(self.screen, color, pos, GRID_WIDTH // 2 - 2)
        # Highlight the last move
        if self.last_move and self.last_move != 'PASS':
            x, y = self.last_move
            pos = (MARGIN + x * GRID_WIDTH, MARGIN + y * GRID_WIDTH)
            pygame.draw.rect(self.screen, (255, 0, 0), (pos[0]-5, pos[1]-5, 10, 10), 2)

    def draw_info_panel(self):
        panel_rect = pygame.Rect(0, WINDOW_HEIGHT - INFO_PANEL_HEIGHT, WINDOW_WIDTH, INFO_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, (240, 240, 240), panel_rect)
        
        text = self.font.render(self.status_text, True, TEXT_COLOR)
        self.screen.blit(text, (20, WINDOW_HEIGHT - INFO_PANEL_HEIGHT + 35))

        if self.game_mode == MODE_MENU:
            self.play_black_button = self.draw_button("Play as Black", 450, WINDOW_HEIGHT - 70, 250, 50)
            self.play_white_button = self.draw_button("Play as White", 750, WINDOW_HEIGHT - 70, 250, 50)
        else:
            self.pass_button = self.draw_button("Pass", 550, WINDOW_HEIGHT - 70, 100, 50)
            self.new_game_button = self.draw_button("New Game", 680, WINDOW_HEIGHT - 70, 150, 50)

    def draw_button(self, text, x, y, w, h):
        mouse_pos = pygame.mouse.get_pos()
        rect = pygame.Rect(x, y, w, h)
        color = BUTTON_HOVER_COLOR if rect.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        text_surf = self.font.render(text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)
        return rect

    def handle_click(self, pos):
        if self.ai_is_thinking: return
        
        if self.game_mode == MODE_MENU:
            if self.play_black_button.collidepoint(pos):
                self.start_new_game(MODE_PLAYER_IS_BLACK)
            elif self.play_white_button.collidepoint(pos):
                self.start_new_game(MODE_PLAYER_IS_WHITE)
        else:
            if self.new_game_button.collidepoint(pos):
                self.game_mode = MODE_MENU
                self.status_text = "Welcome! Select a game mode."
                return
            if self.pass_button.collidepoint(pos) and self.current_player == self.human_player:
                self.handle_pass()
                return

            if self.current_player == self.human_player and not self.game_over:
                x, y = self.get_board_pos(pos)
                if x is not None:
                    self.handle_player_move(x, y)

    def get_board_pos(self, mouse_pos):
        mx, my = mouse_pos
        x = round((mx - MARGIN) / GRID_WIDTH)
        y = round((my - MARGIN) / GRID_WIDTH)
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            return x, y
        return None, None

    def start_new_game(self, mode):
        self.board.reset()
        self.game_mode = mode
        self.current_player = PLAYER_BLACK
        self.last_move = None
        self.pass_count = 0
        self.game_over = False
        self.ai_is_thinking = False

        if mode == MODE_PLAYER_IS_BLACK:
            self.human_player, self.ai_player = PLAYER_BLACK, PLAYER_WHITE
            self.status_text = "You are Black. Your turn."
        elif mode == MODE_PLAYER_IS_WHITE:
            self.human_player, self.ai_player = PLAYER_WHITE, PLAYER_BLACK
            self.status_text = "You are White. AI (Black) is thinking..."
        else: # Back to menu
            self.human_player, self.ai_player = None, None
            self.status_text = "Welcome! Select a game mode."

    def handle_player_move(self, x, y):
        if self.board.is_valid_move(x, y, self.current_player):
            self.board.place_stone(x, y, self.current_player)
            self.last_move = (x, y)
            self.pass_count = 0
            self.switch_player()
        else:
            self.status_text = "Invalid move! Try again."

    def handle_pass(self):
        self.board.pass_turn()
        self.pass_count += 1
        self.last_move = 'PASS'
        print(f"Player {self.current_player} passed. Consecutive passes: {self.pass_count}")
        self.switch_player()

    def handle_ai_move(self):
        self.ai_is_thinking = True
        self.status_text = "AI is thinking..."
        self.draw_and_update()
        
        max_retries = 5
        for attempt in range(max_retries):
            # 1. Convert board to string and get prediction from the model
            input_sequence = board_to_text_representation(self.board, self.ai_player)
            predicted_notation = infer_from_sequence(input_sequence)
            print(f"--- Attempt {attempt+1}/{max_retries} ---")
            print(f"AI model predicted move: '{predicted_notation}'")

            # 2. Handle a "PASS" move
            if predicted_notation and predicted_notation.strip().upper() == 'PASS':
                print("AI chose to pass.")
                self.board.pass_turn()
                self.last_move = 'PASS'
                self.pass_count += 1
                self.ai_is_thinking = False
                self.switch_player()
                return

            # 3. Convert notation to coordinates
            move = from_notation(predicted_notation)
            
            if move:
                x, y = move
                # 4a. Check if the original predicted move is valid
                if self.board.is_valid_move(x, y, self.ai_player):
                    print(f"Move {predicted_notation} is valid. Placing stone.")
                    self.board.place_stone(x, y, self.ai_player)
                    self.last_move = (x, y)
                    self.pass_count = 0
                    self.ai_is_thinking = False
                    self.switch_player()
                    return # Success
                
                # 4b. If the original move is invalid, try mirroring the y-coordinate
                else:
                    print(f"Move {predicted_notation} at ({x},{y}) is invalid. Trying mirrored move as a fallback.")
                    mirrored_y = (BOARD_SIZE - 1) - y
                    mirrored_move = (x, mirrored_y)
                    
                    # Check if the mirrored move is valid
                    if self.board.is_valid_move(mirrored_move[0], mirrored_move[1], self.ai_player):
                        mirrored_notation = to_notation(mirrored_move)
                        print(f"Mirrored move {mirrored_notation} at ({mirrored_move[0]},{mirrored_move[1]}) is valid. Placing stone.")
                        self.board.place_stone(mirrored_move[0], mirrored_move[1], self.ai_player)
                        self.last_move = mirrored_move
                        self.pass_count = 0
                        self.ai_is_thinking = False
                        self.switch_player()
                        return # Success with mirrored move
                    else:
                        # If both original and mirrored moves are invalid, the loop will try again
                        print(f"Mirrored move at ({mirrored_move[0]},{mirrored_move[1]}) is also invalid. Requesting new prediction from model.")

            else:
                print(f"AI returned invalid notation: '{predicted_notation}'. Requesting new prediction from model.")
        
        # If the loop finishes without finding any valid move (original or mirrored)
        print("AI failed to find a valid move after all attempts and fallbacks. AI passes.")
        self.board.pass_turn()
        self.last_move = 'PASS'
        self.pass_count += 1
        self.ai_is_thinking = False
        self.switch_player()
        
    def switch_player(self):
        if self.pass_count >= 2:
            self.status_text = "Game Over (2 consecutive passes)."
            self.game_over = True
            return

        self.current_player = PLAYER_WHITE if self.current_player == PLAYER_BLACK else PLAYER_BLACK
        
        if not self.game_over:
            if self.current_player == self.human_player:
                self.status_text = "Your turn."
            elif self.game_mode != MODE_MENU:
                self.status_text = "AI's turn."

    def draw_and_update(self):
        self.draw_board()
        self.draw_stones()
        self.draw_info_panel()
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            # AI makes a move if it's its turn
            if self.game_mode != MODE_MENU and self.current_player == self.ai_player and not self.game_over and not self.ai_is_thinking:
                self.handle_ai_move()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)

            self.draw_and_update()
            # A small delay to keep the CPU usage down
            pygame.time.wait(10)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    go_board = Board(BOARD_SIZE)
    game = GameUI(go_board)
    game.run()