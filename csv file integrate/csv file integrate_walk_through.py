import pandas as pd
import os


def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if '\u4e00' <= ch <= '\u9fa5':
            return True

First_flag = True
root_dir = r'D:\pjxt\Lamando\results\Comfort'
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if '.csv' in file:
            file_path = os.path.join(dirpath, file)
            print(file_path)
            if First_flag:
                data = pd.DataFrame(pd.read_csv(file_path, encoding='gb18030'))
                name_list = data.keys().tolist()
                for name in name_list:
                    if is_chinese(name):
                        # data.drop(columns=name)
                        del data[name]
                        # print(name + 'in')
                First_flag = False
            else:
                data = data.append(pd.DataFrame(pd.read_csv(file_path, encoding='gb18030')))
                name_list = data.keys().tolist()
                for name in name_list:
                    if is_chinese(name):
                        # data.drop(columns=name)
                        del data[name]
                        # print(name + 'in')
                # for name in data.keys():
                #     if is_chinese(name):
                #         data.drop(columns=name)

output = data.to_csv(r'C:\Users\lvhui\Desktop\output.csv')