from itertools import product

def generate_vocab():
    with open('rwkv_Goose_Go_vocab.txt', 'w', encoding='utf-8') as f:
        # 写入前两行固定内容
        f.write('1 " " 1\n')
        f.write("2 'X' 1\n")
        f.write("3 'B' 1\n")
        f.write("4 'W' 1\n")
        f.write("5 '#' 1\n")        
        f.write("6 '\\n' 1\n")

        # 生成大小写字母组合 Aa 到 Tt
        first_letters = 'ABCDEFGHIJKLMNOPQRS'
        second_letters = 'abcdefghijklmnopqrs'
        index = 7
        
        for first_char in first_letters:
            for second_char in second_letters:
                token = first_char + second_char
                f.write(f"{index} '{token}' 2\n")
                index += 1

        f.write("368 'Black' 5\n")
        f.write("369 'White' 5\n")

if __name__ == "__main__":
    generate_vocab()
    print("词表文件 rwkv_Goose_Go_vocab.txt 已生成完成！")
