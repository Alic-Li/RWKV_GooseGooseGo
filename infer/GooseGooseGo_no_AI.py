import pygame
import sys
import time

BOARD_SIZE = 19
GRID_WIDTH = 60
MARGIN = 60
INFO_PANEL_HEIGHT = 100
WINDOW_WIDTH = 2 * MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH
WINDOW_HEIGHT = 2 * MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH + INFO_PANEL_HEIGHT

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (218, 165, 32) 
LINE_COLOR = (50, 50, 50)
TEXT_COLOR = (10, 10, 10)
BUTTON_COLOR = (100, 100, 200)
BUTTON_HOVER_COLOR = (150, 150, 250)
BUTTON_TEXT_COLOR = WHITE

PLAYER_BLACK = 1
PLAYER_WHITE = 2

MODE_MENU = 0
MODE_PVP = 1 


COORDS_X = "ABCDEFGHIJKLMNOPQRS" 
COORDS_Y = "abcdefghijklmnopqrs" 

def to_notation(point):
    """ (x, y) a1, b2 ... or 'PASS' -> 'X' """
    if point is None:
        return ""
    if point == 'PASS':
        return 'X'
    x, y = point
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return ""
    return COORDS_X[x] + COORDS_Y[y]

def from_notation(notation):
    """ a1, b2 ... to (x, y) """
    if not isinstance(notation, str) or len(notation) != 2:
        return None
    x_char, y_char = notation[0], notation[1]
    if x_char not in COORDS_X or y_char not in COORDS_Y:
        return None
    x = COORDS_X.index(x_char)
    y = COORDS_Y.index(y_char)
    return x, y

class Board:
    def __init__(self, size):
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.history = []
        self.ko_point = None # 用于处理打劫规则

    def reset(self):
        self.grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.history = []
        self.ko_point = None

    def pass_turn(self):
        """Records a 'pass' move in the history."""
        self.history.append('PASS')

    def is_valid_move(self, x, y, player):
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        if self.grid[y][x] != 0:
            return False
        if (x, y) == self.ko_point:
            return False
        self.grid[y][x] = player
        is_suicide = not self._has_liberties((x, y)) and not self._will_capture((x, y), player)
        self.grid[y][x] = 0
        if is_suicide:
            return False
        return True

    def place_stone(self, x, y, player):
        if not self.is_valid_move(x, y, player):
            return []

        self.grid[y][x] = player
        self.history.append((x, y))
        
        captured_stones = self._capture_stones(x, y, player)
        self.ko_point = None
        if len(captured_stones) == 1:
            opponent = 3 - player
            if not self._has_liberties((x, y)):
                captured_x, captured_y = captured_stones[0]
                self.grid[captured_y][captured_x] = opponent 
                if not self._has_liberties((captured_x, captured_y)):
                    self.ko_point = (captured_x, captured_y)
                self.grid[captured_y][captured_x] = 0

        return captured_stones

    def _capture_stones(self, x, y, player):
        captured_stones = []
        opponent = 3 - player
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[ny][nx] == opponent:
                if not self._has_liberties((nx, ny)):
                    group, _ = self._find_group(nx, ny)
                    for stone in group:
                        self.grid[stone[1]][stone[0]] = 0
                        if stone not in captured_stones:
                            captured_stones.append(stone)
        return captured_stones

    def _will_capture(self, point, player):
        x, y = point
        opponent = 3 - player
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[ny][nx] == opponent:
                group, liberties = self._find_group(nx, ny)
                if len(liberties) == 1 and (x, y) in liberties:
                    return True
        return False

    def _find_group(self, x, y):
        player = self.grid[y][x]
        if player == 0:
            return [], []
        
        q = [(x, y)]
        visited = set([(x, y)])
        group = []
        liberties = set()

        while q:
            cx, cy = q.pop(0)
            group.append((cx, cy))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[ny][nx] == 0:
                        liberties.add((nx, ny))
                    elif self.grid[ny][nx] == player and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group, liberties

    def _has_liberties(self, point):
        x, y = point
        _, liberties = self._find_group(x, y)
        return len(liberties) > 0

    def get_history_string(self):
        history_str = " ".join([to_notation(p) for p in self.history])
        if history_str:
            return history_str + " "
        return ""

# --- Pygame UI ---
class GameUI:
    def __init__(self, board):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Go Game") # Changed title
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.board = board
        self.game_mode = MODE_MENU # Start in menu mode
        self.current_player = PLAYER_BLACK
        self.status_text = "Welcome! Select a game mode."
        self.last_move = None
        self.pass_count = 0 # Track consecutive passes for game end

    def draw_board(self):
        self.screen.fill(BOARD_COLOR)
        # Draw lines
        for i in range(BOARD_SIZE):
            start_pos_h = (MARGIN + i * GRID_WIDTH, MARGIN)
            end_pos_h = (MARGIN + i * GRID_WIDTH, MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH)
            pygame.draw.line(self.screen, LINE_COLOR, start_pos_h, end_pos_h)

            start_pos_v = (MARGIN, MARGIN + i * GRID_WIDTH)
            end_pos_v = (MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH, MARGIN + i * GRID_WIDTH)
            pygame.draw.line(self.screen, LINE_COLOR, start_pos_v, end_pos_v)

        # Draw star points
        star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        for p in star_points:
            pos = (MARGIN + p[0] * GRID_WIDTH, MARGIN + p[1] * GRID_WIDTH)
            pygame.draw.circle(self.screen, LINE_COLOR, pos, 5)

        # Draw coordinates
        for i in range(BOARD_SIZE):
            # Horizontal (A-T)
            text = self.small_font.render(COORDS_X[i], True, TEXT_COLOR)
            self.screen.blit(text, (MARGIN + i * GRID_WIDTH - text.get_width() // 2, MARGIN - 30))
            self.screen.blit(text, (MARGIN + i * GRID_WIDTH - text.get_width() // 2, WINDOW_HEIGHT - INFO_PANEL_HEIGHT - MARGIN + 20))
            # Vertical (a-t)
            text = self.small_font.render(COORDS_Y[i], True, TEXT_COLOR)
            self.screen.blit(text, (MARGIN - 30, MARGIN + i * GRID_WIDTH - text.get_height() // 2))
            self.screen.blit(text, (WINDOW_WIDTH - MARGIN + 20, MARGIN + i * GRID_WIDTH - text.get_height() // 2))

    def draw_stones(self):
        for y in range(self.board.size):
            for x in range(self.board.size):
                player = self.board.grid[y][x]
                if player != 0:
                    color = BLACK if player == PLAYER_BLACK else WHITE
                    pos = (MARGIN + x * GRID_WIDTH, MARGIN + y * GRID_WIDTH)
                    pygame.draw.circle(self.screen, color, pos, GRID_WIDTH // 2 - 1)
        
        # Mark last move
        if self.last_move:
            x, y = self.last_move
            pos = (MARGIN + x * GRID_WIDTH, MARGIN + y * GRID_WIDTH)
            # Mark color based on stone color for better visibility
            if self.board.grid[y][x] == PLAYER_BLACK:
                pygame.draw.rect(self.screen, (255, 255, 0), (pos[0]-5, pos[1]-5, 10, 10), 2) # Yellow mark for black stone
            else:
                pygame.draw.rect(self.screen, (255, 0, 0), (pos[0]-5, pos[1]-5, 10, 10), 2) # Red mark for white stone


    def draw_info_panel(self):
        panel_rect = pygame.Rect(0, WINDOW_HEIGHT - INFO_PANEL_HEIGHT, WINDOW_WIDTH, INFO_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, (240, 240, 240), panel_rect)
        
        # Status text
        text = self.font.render(self.status_text, True, TEXT_COLOR)
        self.screen.blit(text, (20, WINDOW_HEIGHT - INFO_PANEL_HEIGHT + 35))

        # Buttons
        # Renamed buttons for clarity in Player vs Player mode
        self.pvp_button = self.draw_button("Player vs Player", 450, WINDOW_HEIGHT - 70, 200, 50)
        self.pass_button = self.draw_button("Pass", 700, WINDOW_HEIGHT - 70, 200, 50) # Added a Pass button
        self.new_game_button = self.draw_button("New Game", 950, WINDOW_HEIGHT - 70, 200, 50)

    def draw_button(self, text, x, y, w, h):
        mouse_pos = pygame.mouse.get_pos()
        rect = pygame.Rect(x, y, w, h)
        
        color = BUTTON_COLOR
        if rect.collidepoint(mouse_pos):
            color = BUTTON_HOVER_COLOR
            
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        
        text_surf = self.font.render(text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)
        return rect

    def handle_click(self, pos):
        if self.game_mode == MODE_MENU:
            if self.pvp_button.collidepoint(pos):
                self.start_new_game(MODE_PVP)
        elif self.game_mode == MODE_PVP:
            if self.new_game_button.collidepoint(pos):
                self.start_new_game(MODE_MENU)
                return
            if self.pass_button.collidepoint(pos): # Handle Pass button
                self.handle_pass()
                return

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
        self.current_player = PLAYER_BLACK # Black always starts
        self.last_move = None
        self.pass_count = 0 # Reset pass count

        if mode == MODE_PVP:
            self.status_text = "Black Player's Turn"
        else: # MODE_MENU
            self.status_text = "Welcome! Select a game mode."


    def handle_player_move(self, x, y):
        if self.board.is_valid_move(x, y, self.current_player):
            self.board.place_stone(x, y, self.current_player)
            self.last_move = (x, y)
            self.pass_count = 0 # Reset pass count on a valid move
            self.switch_player()
        else:
            player_name = "Black" if self.current_player == PLAYER_BLACK else "White"
            self.status_text = f"Invalid move for {player_name}! Try again."


    def handle_pass(self):
        self.board.pass_turn()
        self.pass_count += 1
        self.last_move = None # No physical stone placed for a pass
        self.switch_player()


    def switch_player(self):
        # Check for game end before switching player
        if self.pass_count >= 2:
            self.status_text = "Game Over! Two consecutive PASSes."
            print("Game Over! Two consecutive PASSes.")
            self.game_mode = MODE_MENU # Return to menu or end game state
            return # Prevent further player switching or interaction

        self.current_player = PLAYER_WHITE if self.current_player == PLAYER_BLACK else PLAYER_BLACK
        player_name = "Black" if self.current_player == PLAYER_BLACK else "White"
        self.status_text = f"{player_name} Player's Turn"


    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos) # Always handle clicks

            self.draw_board()
            self.draw_stones()
            self.draw_info_panel()
            
            pygame.display.flip()
            
            # No AI to trigger, so no need for AI-specific logic here
            time.sleep(0.01) # Small delay to prevent CPU overuse

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    go_board = Board(BOARD_SIZE)
    game = GameUI(go_board)
    game.run()