import pygame
import sys
import time
import os
import copy
import types
import torch

os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_JIT_ON"] = "1"
# '1' to compile CUDA kernel (requires compiler), '0' for CPU-friendly version
os.environ["RWKV_CUDA_ON"] = "0" 

########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
# Modified for Go Game Inference
# --- AI BACKEND START ---
########################################################################################################

print("RWKV GooseGooseGo Go Game Inference Model")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# IMPORTANT: Set your model and vocab paths here.
MODEL_PATH = "rwkv-final" # Corrected to include extension
VOCAB_PATH = "data/tokenizer/rwkv_Goose_Go_vocab.txt"


from data.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

args = types.SimpleNamespace()
# Change to "cpu fp32" if you don't have a CUDA-enabled GPU
args.strategy = "cuda fp16" 
args.MODEL_NAME = MODEL_PATH

STATE_NAME = None
# IMPORTANT: Temperature MUST be > 0 for re-inference to work. 
# A value of 0 makes the output deterministic, causing infinite loops on invalid moves.
GEN_TEMP = 0.8
GEN_TOP_P = 0.95 # Using a top_p is also good practice for sampling

print(f"Loading model - {args.MODEL_NAME} with strategy {args.strategy}")
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
tokenizer = TRIE_TOKENIZER(VOCAB_PATH)
model_state = None
init_state = None

if STATE_NAME is not None:
    args = model.args
    state_raw = torch.load(STATE_NAME + '.pth')
    state_init = [None for i in range(args.n_layer * 3)]
    for i in range(args.n_layer):
        dd = model.strategy[i]
        dev = dd.device
        atype = dd.atype    
        state_init[i*3+0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
        state_init[i*3+1] = state_raw[f'blocks.{i}.att.time_state'].transpose(1,2).to(dtype=torch.float, device=dev).requires_grad_(False).contiguous()
        state_init[i*3+2] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
    init_state = copy.deepcopy(state_init)

def reset_model_state():
    """Resets the RNN state. Call this before starting a new game."""
    global model_state
    model_state = copy.deepcopy(init_state) if init_state is not None else None
    print("AI model state has been reset for a new game.")

def predict_go_move(input_move_notation):
    """
    Predicts the NEXT TOKEN based on the previous single move/token.
    Manages the RNN state internally.
    """
    global model_state
    
    if input_move_notation is None:
        tokens = [0]
    else:
        tokens = tokenizer.encode(input_move_notation)
        if not tokens:
            print(f"Warning: Could not encode move '{input_move_notation}'. Using a default token.")
            tokens = [0]

    out, model_state_new = model.forward(tokens, model_state)
    model_state = model_state_new # Update the global state
    
    token = pipeline.sample_logits(out, temperature=GEN_TEMP, top_p=GEN_TOP_P)
    
    return tokenizer.decode([token])

# --- AI BACKEND END ---
########################################################################################################


# --- PYGAME FRONTEND START ---
BOARD_SIZE = 19
GRID_WIDTH = 60
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
COORDS_Y = "abcdefghijklmnopqrs" 

def to_notation(point):
    if point is None: return None
    if point == 'PASS': return 'X'
    x, y = point
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE): return ""
    return COORDS_X[x] + COORDS_Y[y]

def from_notation(notation):
    if not isinstance(notation, str) or len(notation) != 2: return None
    notation = notation.strip()
    if len(notation) != 2: return None
    x_char, y_char = notation[0].upper(), notation[1].lower()
    if x_char not in COORDS_X or y_char not in COORDS_Y: return None
    x = COORDS_X.index(x_char)
    y = COORDS_Y.index(y_char)
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
        
        self.grid[y][x] = player
        is_suicide = not self._has_liberties((x, y)) and not self._will_capture((x, y), player)
        self.grid[y][x] = 0
        if is_suicide: return False
        
        return True

    def place_stone(self, x, y, player):
        if not self.is_valid_move(x, y, player): return []

        self.grid[y][x] = player
        self.history.append((x, y))
        
        captured_stones = self._capture_stones(x, y, player)
        
        self.ko_point = None
        if len(captured_stones) == 1:
            captured_x, captured_y = captured_stones[0]
            # Check if the move that resulted in a single capture was itself part of a group with no liberties (a ko-like situation)
            if not self._has_liberties((x, y)):
                 # Simulate placing the captured stone back to see if it would have no liberties (a true ko)
                 self.grid[captured_y][captured_x] = 3 - player
                 if not self._has_liberties((captured_x, captured_y)):
                      self.ko_point = (captured_x, captured_y)
                 self.grid[captured_y][captured_x] = 0 # Revert simulation

        return captured_stones

    def _capture_stones(self, x, y, player):
        captured_stones = []
        opponent = 3 - player
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[ny][nx] == opponent:
                if not self._has_liberties((nx, ny)):
                    group, _ = self._find_group(nx, ny)
                    for stone_x, stone_y in group:
                        self.grid[stone_y][stone_x] = 0
                        if (stone_x, stone_y) not in captured_stones:
                            captured_stones.append((stone_x, stone_y))
        return captured_stones

    def _will_capture(self, point, player):
        x, y = point
        self.grid[y][x] = player
        opponent = 3 - player
        captured_something = False
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[ny][nx] == opponent:
                if not self._has_liberties((nx, ny)):
                    captured_something = True
                    break
        self.grid[y][x] = 0
        return captured_something

    def _find_group(self, x, y):
        player = self.grid[y][x]
        if player == 0: return [], []
        q, visited, group, liberties = [(x, y)], set([(x, y)]), [], set()
        while q:
            cx, cy = q.pop(0)
            group.append((cx, cy))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[ny][nx] == 0: liberties.add((nx, ny))
                    elif self.grid[ny][nx] == player and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group, liberties

    def _has_liberties(self, point):
        x, y = point
        _, liberties = self._find_group(x, y)
        return len(liberties) > 0


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
            start_pos_h = (MARGIN + i * GRID_WIDTH, MARGIN)
            end_pos_h = (MARGIN + i * GRID_WIDTH, MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH)
            pygame.draw.line(self.screen, LINE_COLOR, start_pos_h, end_pos_h)
            start_pos_v = (MARGIN, MARGIN + i * GRID_WIDTH)
            end_pos_v = (MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH, MARGIN + i * GRID_WIDTH)
            pygame.draw.line(self.screen, LINE_COLOR, start_pos_v, end_pos_v)
        star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        for p in star_points:
            pos = (MARGIN + p[0] * GRID_WIDTH, MARGIN + p[1] * GRID_WIDTH)
            pygame.draw.circle(self.screen, LINE_COLOR, pos, 5)
        for i in range(BOARD_SIZE):
            text = self.small_font.render(COORDS_X[i], True, TEXT_COLOR)
            self.screen.blit(text, (MARGIN + i * GRID_WIDTH - text.get_width() // 2, MARGIN - 30))
            text = self.small_font.render(str(i + 1), True, TEXT_COLOR)
            self.screen.blit(text, (MARGIN - 30, MARGIN + i * GRID_WIDTH - text.get_height() // 2))

    def draw_stones(self):
        for y in range(self.board.size):
            for x in range(self.board.size):
                player = self.board.grid[y][x]
                if player != 0:
                    color = BLACK_COLOR if player == PLAYER_BLACK else WHITE_COLOR
                    pos = (MARGIN + x * GRID_WIDTH, MARGIN + y * GRID_WIDTH)
                    pygame.draw.circle(self.screen, color, pos, GRID_WIDTH // 2 - 2)
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
            self.play_black_button = self.draw_button("Play as Black (vs AI)", 150, WINDOW_HEIGHT - 70, 250, 50)
            self.play_white_button = self.draw_button("Play as White (vs AI)", 450, WINDOW_HEIGHT - 70, 250, 50)
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
                self.start_new_game(MODE_MENU)
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
        reset_model_state()
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
        else:
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

    # <--- NEW, REFINED LOGIC
    def handle_ai_move(self):
        """
        Handles AI's turn with a re-inference loop for invalid moves.
        """
        global model_state
        self.ai_is_thinking = True
        self.status_text = "AI is thinking..."
        self.draw_and_update()
        
        MAX_REINFERENCE_ATTEMPTS = 15 # Max number of times AI tries to find a valid move in a single turn
        
        # This is the main loop for re-inferring if a move is invalid.
        for attempt in range(MAX_REINFERENCE_ATTEMPTS):
            # IMPORTANT: Save the state *before* this attempt.
            state_before_this_attempt = copy.deepcopy(model_state)
            
            # Determine the input for the AI's first prediction in this attempt.
            last_input_for_ai = to_notation(self.board.history[-1] if self.board.history else None)
            
            # Inner loop to construct a full coordinate (e.g., 'P' + 'p' -> "Pp")
            current_attempt_str = ""
            for _ in range(4): # Try to build a move up to 4 tokens long
                next_token = predict_go_move(last_input_for_ai)

                if next_token is None or next_token == '\ufffd' or next_token.strip().upper() == 'X':
                    current_attempt_str = "X" # Treat as a pass
                    break
                
                current_attempt_str += next_token.strip()

                if from_notation(current_attempt_str):
                    break # Successfully formed a 2-char coordinate
                
                # Use the generated token as the next input
                last_input_for_ai = next_token

            # Now, validate the constructed move notation
            final_move_notation = current_attempt_str
            print(f"Attempt {attempt + 1}: AI constructed move '{final_move_notation}'")

            if final_move_notation.strip().upper() == 'X':
                print("AI chose to pass.")
                self.board.pass_turn()
                self.last_move = 'PASS'
                self.pass_count += 1
                self.ai_is_thinking = False
                self.switch_player()
                return # Exit successfully after passing

            move = from_notation(final_move_notation)
            if move and self.board.is_valid_move(move[0], move[1], self.ai_player):
                print(f"Move '{final_move_notation}' is valid. Placing stone.")
                self.board.place_stone(move[0], move[1], self.ai_player)
                self.last_move = move
                self.pass_count = 0
                self.ai_is_thinking = False
                self.switch_player()
                return # Exit successfully after a valid move

            # If the move was invalid, restore the state and try again.
            print(f"Move '{final_move_notation}' is invalid. Re-inferring...")
            model_state = state_before_this_attempt
            self.draw_and_update() # Keep UI responsive

        # If all attempts fail, the AI passes its turn.
        print(f"AI failed to find a valid move after {MAX_REINFERENCE_ATTEMPTS} attempts. Passing.")
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
            if self.game_mode != MODE_MENU and self.current_player == self.ai_player and not self.game_over and not self.ai_is_thinking:
                self.handle_ai_move()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)

            self.draw_and_update()
            time.sleep(0.01)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    go_board = Board(BOARD_SIZE)
    game = GameUI(go_board)
    game.run()