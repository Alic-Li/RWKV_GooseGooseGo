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
    将围棋数据集转换为每步独立保存的格式，并实现吃子逻辑。
    每个回合一行，格式为：
    [上一步坐标][颜色token]\n[当前棋盘状态]\n[当前落子坐标][颜色token]
    或者如果是第一步则没有前缀。

    Args:
        input_file (str): 输入的JSONL文件名。
        output_file (str): 输出的JSONL文件名。
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, total=total_lines, desc="Processing"):
            data = json.loads(line)
            move_pairs = data['text'].split()

            board_size = 19
            board = [['#' for _ in range(board_size)] for _ in range(board_size)]

            prev_move_coord = None
            prev_color_token = None

            def process_a_single_move(move_coord, is_black):
                nonlocal board, prev_move_coord, prev_color_token

                # 解析坐标
                col = ord(move_coord[0].lower()) - ord('a')
                row = ord(move_coord[1].lower()) - ord('a')

                if not (0 <= row < board_size and 0 <= col < board_size):
                    return  # 无效坐标，跳过

                player_char = 'B' if is_black else 'W'
                opponent_char = 'W' if is_black else 'B'
                color_token = "Black" if is_black else "White"

                # 构造当前棋盘状态字符串
                board_representation = "\n".join(["".join(row) for row in board])

                # 构造输出内容
                if prev_move_coord is not None and prev_color_token is not None:
                    prefix = f"{prev_color_token}{prev_move_coord}\n"
                else:
                    prefix = ""

                output_text = f"{prefix}{board_representation}\n{color_token}{move_coord}"

                # 写入这一回合的内容
                f_out.write(json.dumps({"text": output_text}) + '\n')

                # 落子
                board[row][col] = player_char

                # 检查并移除被吃的对方棋子
                for nr, nc in get_neighbors(row, col, board_size):
                    if board[nr][nc] == opponent_char:
                        opponent_group, group_liberties = find_group(nr, nc, board)
                        if not group_liberties:
                            for gr, gc in opponent_group:
                                board[gr][gc] = '#'

                # 更新上一步信息
                prev_move_coord = move_coord
                prev_color_token = color_token

            # 遍历所有移动对或特殊指令
            for item in move_pairs:
                if len(item) == 4 and item.isalpha():
                    black_move = item[:2]
                    white_move = item[2:]

                    process_a_single_move(black_move, is_black=True)
                    process_a_single_move(white_move, is_black=False)

                elif 'X' in item:
                    # 特殊指令处理：模拟落子，但不改变颜色
                    board_representation = "\n".join(["".join(row) for row in board])

                    if prev_move_coord is not None and prev_color_token is not None:
                        prefix = f"{prev_color_token}{prev_move_coord}\n"
                    else:
                        prefix = ""

                    output_text = f"{prefix}{board_representation}\n{item}"

                    f_out.write(json.dumps({"text": output_text}) + '\n')

                    # 不更新 prev_move_coord 和 prev_color_token


# 运行转换函数
convert_go_dataset(
    input_file="/home/rwkv/alic-li/RWKV_GooseGooseGo/data/dataset_cleaned.jsonl",
    output_file="/home/rwkv/alic-li/RWKV_GooseGooseGo/data/go_capture_simulation_output.jsonl"
)

print("数据转换完成，并已保存到 go_capture_simulation_output.jsonl 文件中。")