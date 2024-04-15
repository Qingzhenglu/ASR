import os

ROOT_DATA = r"data_thchs30/data"

data_list = []

for a, b, files in os.walk(ROOT_DATA):
    for i in range(len(files)):
        temp = ""
        if files[i].endswith('wav'):
            temp += files[i] + '\t'
        if files[i].endswith('trn'):
            with open(os.path.join(a, files[i]), 'r', encoding='utf-8') as f:
                f_ = f.readlines()
                for j in range(2):
                    if j == 0:
                        f_[j] = f_[j].replace(' ', '').replace('\n', '\t')
                    temp += f_[j]
        data_list.append(temp)


with open('data.txt', 'w', encoding='utf-8') as file:
    for v in data_list:
        file.write(str(v))

with open('data.txt', 'r', encoding='utf-8') as file:
    f_ = file.readlines()
    for i in range(10):
        for j in range(3):
            print(f_[i].split('\t')[j])
    print('语音数量:', len(f_))
