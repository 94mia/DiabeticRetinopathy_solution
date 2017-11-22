import codecs


def ReadFile(filePath, encoding):
    with codecs.open(filePath, "r", encoding) as f:
        return f.read()


def WriteFile(filePath, u, encoding):
    with codecs.open(filePath, "w", encoding) as f:
        f.write(u)


'''''
定义GBK_2_UTF8方法，用于转换文件存储编码
'''


def GBK_2_UTF8(src, dst):
    content = ReadFile(src, encoding='gbk')
    WriteFile(dst, content, encoding='utf_8')


'''''
qyx.csv文件使用GBK编码存储，现在将其转为UTF_8存储
'''
src = 'case.csv'
dst = 'case_utf8.csv'
GBK_2_UTF8(src, dst)

import pandas as pd

'''''
读取转换存储格式后的文件
'''
path = 'case_utf8.csv'
data = pd.read_csv(path, )
data.head()

df = pd.DataFrame.from_csv(path)


# index:8 病例诊断
bingli_list = []
for index, row in df.iterrows():
    bingli_list.append(row[9])

ill_case = '老年性'

bingli_list = bingli_list[2:]

import jieba

dict = {}

list_age = []

for index in bingli_list:
    if not isinstance(index, str):
        continue
    if ill_case in index:
        list_age.append(index)
    res = list(jieba.cut(index))
    for i in res:
        if i in dict:
            dict[i] += 1
        else:
            dict[i] = 1


dict_cut = {}
dict = {key:value for key,value in dict.items() if value > 50}

print(dict)
print(list_age)
