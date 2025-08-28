import json
from tqdm import tqdm

def get_neighbors(row, col, board_size):
    """获取一个点的所有有效邻居坐标。"""
    neighbors = []
    if row > 0:
        neighbors.append((row - 1, col))
    if row < board_size - 1:
        neighbors.append((row + 1, col))
    if col > 0:
        neighbors.append((row, col - 1))
    if col < board_size - 1:
        neighbors.append((row, col + 1))
    return neighbors

def find_group(row, col, board):
    """
    找到与(row, col)相连的整个棋子群组及其气。
    使用广度优先搜索 (BFS)。
    """
    board_size = len(board)
    stone_color = board[row][col]
    if stone_color == '#':
        return set(), set()

    group_stones = set()
    liberties = set()
    q = [(row, col)]
    visited = set([(row, col)])

    while q:
        r, c = q.pop(0)
        group_stones.add((r, c))

        for nr, nc in get_neighbors(r, c, board_size):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                neighbor_char = board[nr][nc]
                if neighbor_char == '#':
                    liberties.add((nr, nc))
                elif neighbor_char == stone_color:
                    q.append((nr, nc))
    
    return group_stones, liberties

def convert_go_dataset(input_file='input.jsonl', output_file='output.jsonl'):
    """
    将围棋数据集转换为包含每步棋盘状态的格式，并实现吃子逻辑。
    (已修正解析逻辑)

    Args:
        input_file (str): 输入的JSONL文件名。
        output_file (str): 输出的JSONL文件名。
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, total=total_lines, desc="Processing"):
            data = json.loads(line)
            # .split() 会按空格分割棋谱
            move_pairs = data['text'].split()

            board_size = 19
            board = [['#' for _ in range(board_size)] for _ in range(board_size)]
            output_text_parts = []
            
            # 定义一个内部函数来处理单步棋，避免代码重复
            def process_a_single_move(move_coord, is_black):
                nonlocal board # 允许函数修改外部的 board 变量
                
                # 1. 生成落子前的棋盘状态
                board_representation = "\n".join(["".join(row) for row in board])

                # 2. 解析坐标并落子
                col = ord(move_coord[0].lower()) - ord('a')
                row = ord(move_coord[1].lower()) - ord('a')

                if not (0 <= row < board_size and 0 <= col < board_size):
                    return # 无效坐标，跳过

                player_char = 'B' if is_black else 'W'
                opponent_char = 'W' if is_black else 'B'
                color_token = "Black" if is_black else "White"

                board[row][col] = player_char

                # 3. 检查并移除被吃的对方棋子
                for nr, nc in get_neighbors(row, col, board_size):
                    if board[nr][nc] == opponent_char:
                        opponent_group, group_liberties = find_group(nr, nc, board)
                        if not group_liberties:
                            for gr, gc in opponent_group:
                                board[gr][gc] = '#'
                
                # 4. 记录这一步的结果
                output_text_parts.append(f"{board_representation}\n{move_coord}{color_token}")


            # 遍历所有移动对或特殊指令
            for item in move_pairs:
                if len(item) == 4 and item.isalpha():
                    # --- 处理一个标准的四字符移动对，例如 "PcCp" ---
                    black_move = item[:2]
                    white_move = item[2:]
                    
                    # 处理黑棋
                    process_a_single_move(black_move, is_black=True)
                    
                    # 处理白棋
                    process_a_single_move(white_move, is_black=False)

                elif 'X' in item:
                    # --- 处理特殊指令，例如 "HmX" ---
                    board_representation = "\n".join(["".join(row) for row in board])
                    # 根据用户要求，直接输出坐标和X
                    output_text_parts.append(f"{board_representation}\n{item}")

            # 将所有步的输出合并为一个字符串并写入文件
            final_text = "".join(output_text_parts)
            f_out.write(json.dumps({"text": final_text}) + '\n')


# 运行转换函数
convert_go_dataset(input_file="/mnt/69043a6d-b152-4bd1-be10-e1130af6487f/dataset_cleaned.jsonl", output_file="/mnt/3f7ab3b2-e663-407a-831c-ee4789165577/go_capture_simulation_output.jsonl")

print("数据转换完成，并已保存到 go_capture_simulation_output.jsonl 文件中（已包含吃子逻辑）。")