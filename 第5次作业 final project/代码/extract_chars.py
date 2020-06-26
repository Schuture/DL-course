''' 提取出训练集所有出现过的字符 '''
import pathlib

S = set()

data_path = '/root/16307110435_zjw/deeplearn/final/data/train/'
p = pathlib.Path(data_path) / 'gt'
for gt in p.glob('*.txt'):
    with gt.open(mode='r') as f:
        for line in f: # 一行是“x1,y1,x2,y2,x3,y3,x4,y4,语言,内容”
            text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',')
            transcript = text[9] # 只要内容，不管语言
            for s in transcript: # 将出现过的字符逐一加入集合
                S.add(s)
                
S = ''.join(sorted(list(S)))

with open('/root/16307110435_zjw/deeplearn/final/FOTS-master/utils/common_str.py',
          'a', encoding='utf-8') as f:
    f.write('DL_str = \'{}\''.format(S))