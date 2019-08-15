import pandas as pd
import os

root_dir = r'E:\个人文档\上汽\车型试验\凌渡 vs IP31\Lamando_14T_DCT_L17SBV002_Banchmark_65%_20190617_results\Comfort\TipInOut'
file_list = os.listdir(root_dir)
First_flag = True

for file in file_list:
    print(file)
    file_path = os.path.join(root_dir, file)
    if First_flag:
        data = pd.DataFrame(pd.read_csv(file_path, encoding='gb18030'))
        First_flag = False
    else:
        data = data.append(pd.DataFrame(pd.read_csv(file_path, encoding='gb18030')))

output = data.to_csv(r'C:\Users\lvhui\Desktop\TipInOut.csv')