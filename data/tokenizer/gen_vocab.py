def generate_vocab():
    with open('rwkv_Goose_Go_vocab.txt', 'w', encoding='utf-8') as f:
        # 写入前两行固定内容
        f.write('1 " " 1\n')
        f.write("2 'X' 1\n")
        
        # 生成大小写字母组合 Aa 到 Tt
        # 第一个字母大写 A-T，第二个字母小写 a-t
        first_letters = 'ABCDEFGHIJKLMNOPQRST'  # 大写字母 A-T
        second_letters = 'abcdefghijklmnopqrst'  # 小写字母 a-t
        index = 3  # 从索引3开始
        
        for first_char in first_letters:
            for second_char in second_letters:
                token = first_char + second_char
                f.write(f"{index} '{token}' 2\n")
                index += 1

if __name__ == "__main__":
    generate_vocab()
    print("词表文件 rwkv_Goose_Go_vocab.txt 已生成完成！")
    print("总共生成了 2 (固定) + 361 (组合) = 363 个token")
    print("组合范围从 Aa 到 Tt")