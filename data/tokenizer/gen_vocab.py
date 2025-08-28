from itertools import product

def generate_vocab():
    with open('rwkv_Goose_Go_vocab.txt', 'w', encoding='utf-8') as f:
        # 写入前两行固定内容
        f.write('1 " " 1\n')
        f.write("2 'X' 1\n")
        f.write("11 '\n' 1")
        
        # 生成大小写字母组合 Aa 到 Tt
        first_letters = 'ABCDEFGHIJKLMNOPQRS'
        second_letters = 'abcdefghijklmnopqrs'
        index = 4
        
        for first_char in first_letters:
            for second_char in second_letters:
                token = first_char + second_char
                f.write(f"{index} '{token}' 2\n")
                index += 1

        # 加入围棋棋盘字符组合：#、B、W 的长度1~4组合
        go_chars = ['#', 'B', 'W']
        
        for n in range(1, 7):  # 长度1到6
            for combo in product(go_chars, repeat=n):
                token = ''.join(combo)
                f.write(f"{index} '{token}' {n}\n")
                index += 1

if __name__ == "__main__":
    generate_vocab()
    print("词表文件 rwkv_Goose_Go_vocab.txt 已生成完成！")
    print("总共生成了 2 (固定) + 361 (字母组合) + 120 (围棋组合) = 483 个token")
