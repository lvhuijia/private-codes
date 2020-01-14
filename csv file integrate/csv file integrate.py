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


root_dir = r'E:\个人文档\上汽\csv'
file_list = os.listdir(root_dir)
file_list_order = [int(name.split('_')[-1][0:-8]) for name in file_list]
file_list_ordered = []
for order in range(1, len(file_list)+1):
    file_list_ordered.append(file_list[file_list_order.index(order)])
First_flag = True

for file in file_list_ordered:
    print(file)
    file_path = os.path.join(root_dir, file)
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