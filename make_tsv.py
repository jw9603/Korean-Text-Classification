import csv

input_file = './ratings.txt'
output_file = './review.sorted.refine.tsv'

# 전각 문자와 반각 문자를 매핑하는 테이블
fullwidth_to_halfwidth = str.maketrans(
    '０１２３４５６７８９＇＂（）［］｛｝＠＃＄％＾＆＊＋＝｜＜＞？／．',
    '0123456789\'"()[]{}@#$%^&*+=|<>?/.' 
)

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

output_lines = []

# 컬럼 제목은 제외하고
for line in lines[1:]:
    parts = line.strip().split('\t')
    label = 'positive'if parts[2] == '1' else 'negative'
    if not parts[1]: 
        continue
    parts[1] = parts[1].translate(fullwidth_to_halfwidth)
    output_lines.append([label, parts[1]])

output_lines.sort(key=lambda x:x[1])

with open(output_file, 'w',newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    # 데이터 기록
    writer.writerows(output_lines)
    
print(output_file[0])