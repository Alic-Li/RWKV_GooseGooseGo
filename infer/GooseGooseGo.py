
import pygame
import sys
import time
import random
from threading import Thread, Lock

# Make sure rwkv_go_infer_model.py is in the same directory or accessible
try:
    from rwkv_go_infer_model import predict_go_move
except ImportError:
    print("错误：无法导入 'predict_go_move'。")
    print("请确保 'rwkv_go_infer_model.py' 文件在同一个目录下，并且所有依赖项都已安装。")
    sys.exit(1)

# --- 游戏常量 ---
# 棋盘尺寸
BOARD_SIZE = 19
# 棋盘和窗口的视觉参数
GRID_WIDTH = 60
MARGIN = 60
INFO_PANEL_HEIGHT = 100
WINDOW_WIDTH = 2 * MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH
WINDOW_HEIGHT = 2 * MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH + INFO_PANEL_HEIGHT

# 颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (218, 165, 32) # SADDLEBROWN
LINE_COLOR = (50, 50, 50)
TEXT_COLOR = (10, 10, 10)
BUTTON_COLOR = (100, 100, 200)
BUTTON_HOVER_COLOR = (150, 150, 250)
BUTTON_TEXT_COLOR = WHITE

# 游戏状态
PLAYER_BLACK = 1
PLAYER_WHITE = 2

# 游戏模式
MODE_MENU = 0
MODE_PVA = 1 # Player vs AI
MODE_AVA = 2 # AI vs AI

# --- 坐标转换 ---
COORDS_X = "ABCDEFGHJKLMNOPQRST" # 排除 'I'
COORDS_Y = "abcdefghjklmnopqrst" # 排除 'i'

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

# --- 核心游戏逻辑 ---
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
        # 检查是否在棋盘内
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        # 检查是否已经有棋子
        if self.grid[y][x] != 0:
            return False
        # 检查是否是打劫禁手
        if (x, y) == self.ko_point:
            return False
        # 检查是否是自杀禁手
        # 临时放置棋子以检查
        self.grid[y][x] = player
        is_suicide = not self._has_liberties((x, y)) and not self._will_capture((x, y), player)
        # 撤销临时放置
        self.grid[y][x] = 0
        if is_suicide:
            return False
        return True

    def place_stone(self, x, y, player):
        if not self.is_valid_move(x, y, player):
            return []

        self.grid[y][x] = player
        self.history.append((x, y))
        
        # 处理吃子
        captured_stones = self._capture_stones(x, y, player)
        
        # 处理打劫
        self.ko_point = None
        if len(captured_stones) == 1:
            # 检查是否形成了打劫局面
            opponent = 3 - player
            # 如果这个落子吃掉了一个子，并且这个落子点没有其他气了
            if not self._has_liberties((x, y)):
                 # 检查被吃掉的子是否也只吃掉了我们这一个子
                captured_x, captured_y = captured_stones[0]
                self.grid[captured_y][captured_x] = opponent # 临时恢复被吃的子
                if not self._has_liberties((captured_x, captured_y)):
                    self.ko_point = (captured_x, captured_y)
                self.grid[captured_y][captured_x] = 0 # 再次移除

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
        """检查在(x,y)落子是否会吃掉对方的子"""
        x, y = point
        opponent = 3 - player
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[ny][nx] == opponent:
                # 检查对方的这个棋子或棋群是否只有一口气（就是(x,y)这一点）
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

def get_special_substr(history_str):
    if len(history_str) < 2:
        return ''  # 长度不够，无法获取倒数第二字符
    second_last = history_str[-2]  # 倒数第二个字符
    third_last = history_str[-3] if len(history_str) >= 3 else ''  # 倒数第三个字符
    if second_last == 'X':
        return 'X'
    else:
        return third_last + second_last

# --- Pygame UI ---
class GameUI:
    def __init__(self, board):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("RWKV Goose Goose Go🪿")
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.board = board
        self.game_mode = MODE_MENU
        self.current_player = PLAYER_BLACK
        self.status_text = "Welcome to RWKV GooseGooseGo! "
        self.ai_is_thinking = False
        self.ai_lock = Lock()
        self.last_move = None

    def draw_board(self):
        self.screen.fill(BOARD_COLOR)
        # 画线
        for i in range(BOARD_SIZE):
            start_pos_h = (MARGIN + i * GRID_WIDTH, MARGIN)
            end_pos_h = (MARGIN + i * GRID_WIDTH, MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH)
            pygame.draw.line(self.screen, LINE_COLOR, start_pos_h, end_pos_h)

            start_pos_v = (MARGIN, MARGIN + i * GRID_WIDTH)
            end_pos_v = (MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH, MARGIN + i * GRID_WIDTH)
            pygame.draw.line(self.screen, LINE_COLOR, start_pos_v, end_pos_v)

        # 画星位
        star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        for p in star_points:
            pos = (MARGIN + p[0] * GRID_WIDTH, MARGIN + p[1] * GRID_WIDTH)
            pygame.draw.circle(self.screen, LINE_COLOR, pos, 5)

        # 画坐标
        for i in range(BOARD_SIZE):
            # 横坐标 (A-T)
            text = self.small_font.render(COORDS_X[i], True, TEXT_COLOR)
            self.screen.blit(text, (MARGIN + i * GRID_WIDTH - text.get_width() // 2, MARGIN - 30))
            self.screen.blit(text, (MARGIN + i * GRID_WIDTH - text.get_width() // 2, WINDOW_HEIGHT - INFO_PANEL_HEIGHT - MARGIN + 20))
            # 纵坐标 (a-t)
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
        
        # 标记上一手棋
        if self.last_move:
            x, y = self.last_move
            pos = (MARGIN + x * GRID_WIDTH, MARGIN + y * GRID_WIDTH)
            pygame.draw.rect(self.screen, (255, 0, 0), (pos[0]-5, pos[1]-5, 10, 10), 2)


    def draw_info_panel(self):
        panel_rect = pygame.Rect(0, WINDOW_HEIGHT - INFO_PANEL_HEIGHT, WINDOW_WIDTH, INFO_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, (240, 240, 240), panel_rect)
        
        # 状态文本
        text = self.font.render(self.status_text, True, TEXT_COLOR)
        self.screen.blit(text, (20, WINDOW_HEIGHT - INFO_PANEL_HEIGHT + 35))

        # 按钮
        self.pva_button = self.draw_button("Play With Goose", 450, WINDOW_HEIGHT - 70, 200, 50)
        self.ava_button = self.draw_button("Goose SelfPlay", 700, WINDOW_HEIGHT - 70, 200, 50)
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
            if self.pva_button.collidepoint(pos):
                self.start_new_game(MODE_PVA)
            elif self.ava_button.collidepoint(pos):
                self.start_new_game(MODE_AVA)
        elif self.game_mode == MODE_PVA:
            if self.new_game_button.collidepoint(pos):
                self.start_new_game(MODE_MENU)
                return
            
            if self.current_player == PLAYER_BLACK and not self.ai_is_thinking:
                x, y = self.get_board_pos(pos)
                if x is not None:
                    self.handle_player_move(x, y)
        elif self.game_mode == MODE_AVA:
             if self.new_game_button.collidepoint(pos):
                self.start_new_game(MODE_MENU)
                return

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
        if mode == MODE_PVA:
            self.status_text = "Playing with Goose"
        elif mode == MODE_AVA:
            self.status_text = "Goose Self Playing"
            self.trigger_ai_move()
        else: # MODE_MENU
            self.status_text = ""


    def handle_player_move(self, x, y):
        if self.board.is_valid_move(x, y, self.current_player):
            self.board.place_stone(x, y, self.current_player)
            self.last_move = (x, y)
            self.switch_player()
            if self.game_mode == MODE_PVA:
                self.trigger_ai_move()

    def switch_player(self):
        self.current_player = PLAYER_WHITE if self.current_player == PLAYER_BLACK else PLAYER_BLACK
        player_name = "Black" if self.current_player == PLAYER_BLACK else "white"
        
        if self.game_mode == MODE_PVA:
            self.status_text = "AI Thinking" if self.current_player == PLAYER_WHITE else "Pleae"
        elif self.game_mode == MODE_AVA:
            self.status_text = f"AI({player_name})Thinking"

    def trigger_ai_move(self):
        if self.ai_is_thinking:
            return
        self.ai_is_thinking = True
        # 在新线程中运行AI，避免UI卡死
        thread = Thread(target=self.run_ai_logic)
        thread.start()

    def run_ai_logic(self):
        with self.ai_lock:
            history_str = self.board.get_history_string()

            # 如果是第一手棋，随机选择一个位置
            if not history_str:
                print("AI is making a random first move.")
                while True:
                    x = random.randint(0, self.board.size - 1)
                    y = random.randint(0, self.board.size - 1)
                    if self.board.is_valid_move(x, y, self.current_player):
                        self.board.place_stone(x, y, self.current_player)
                        self.last_move = (x, y)
                        self.switch_player()
                        break
            else:
                # AI落子循环，直到找到有效位置
                while True:
                    print(f"AI is thinking... History: '{history_str}'")
                    str = get_special_substr(history_str)
                    ai_move_notation = predict_go_move(str)
                    print(f"AI predicted move: '{ai_move_notation}'")

                    # 纠正模型可能输出的无效字符 'I' 或 'i'
                    if ai_move_notation:
                        original_move = ai_move_notation
                        ai_move_notation = ai_move_notation.replace('I', 'J').replace('i', 'j')
                        if ai_move_notation != original_move:
                            print(f"已将AI落子从 '{original_move}' 修正为 '{ai_move_notation}'")

                    if not ai_move_notation:
                        print("AI未能预测有效落子, 将重试。")
                        # time.sleep(0.5)
                        continue

                    point = from_notation(ai_move_notation)
                    if point and self.board.is_valid_move(point[0], point[1], self.current_player):
                        x, y = point
                        self.board.place_stone(x, y, self.current_player)
                        self.last_move = (x, y)
                        self.switch_player()
                        break # Found a valid move, exit loop.
                    else:
                        # If the move is invalid (or '�'), treat it as a pass and end the turn.
                        print(f"AI's predicted move '{ai_move_notation}' is invalid. Treating as a pass.")
                        self.board.pass_turn()
                        self.last_move = None
                        self.switch_player()
                        break # End turn, exit loop.

            self.ai_is_thinking = False
            # 如果是AI vs AI模式，触发下一个AI
            if self.game_mode == MODE_AVA:
                self.trigger_ai_move()


    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.ai_is_thinking:
                        self.handle_click(event.pos)

            self.draw_board()
            self.draw_stones()
            self.draw_info_panel()
            
            pygame.display.flip()
            
            # 在AI自弈模式下，如果非思考状态，则触发AI
            if self.game_mode == MODE_AVA and not self.ai_is_thinking:
                self.trigger_ai_move()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    go_board = Board(BOARD_SIZE)
    game = GameUI(go_board)
    game.run()
