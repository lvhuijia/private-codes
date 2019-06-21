import pandas as pd
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
import os
import xlsxwriter

fs = 20
st = []
snd = []
ped_exist = []
ped_index = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 71, 80, 90]  # 100舍去，因为100%油门对应的稳态力和位移有多值，不唯一
file_path = r'C:\Users\吕惠加\Desktop\20190228_152601(417_1)_IP32.csv'
# file_path = r'C:\Users\吕惠加\Desktop\20190228_160939(420_4)_IP34.csv'
data = pd.read_csv(file_path, encoding='GB18030')
ped_data = data['AccelActuPosHSC1']
# ped_data = data['EPTAccelActuPosHSC1']
dis_data = data['disp']
force_data = data['MSADMM_PedalForce']

for ii in range(0, len(ped_index)):
    exam1 = np.where(abs(ped_data - ped_index[ii]) <= 1)
    if exam1[0].shape[0] <= 0:
        continue
    st1 = []
    snd1 = []
    for k, g in groupby(enumerate(np.array(exam1[0])),  # 此循环获得每一个连续符合exam条件的数据段的始末点index，即满足车速和pedal要求的所有点中，连续的点分为一组，取每组的头和末
                        lambda x: x[1] - x[0]):
        oo = [v for i, v in g]
        st1.append(oo[0])
        snd1.append(oo[-1])

    exam = np.array([snd1[ii] - st1[ii] for ii in range(len(st1))])
    len1 = 10
    exam4 = np.where(exam > len1 * fs)
    if exam4[0].shape[0] <= 0:
        continue

    st.append(st1[exam4[0][0]]+2*fs)
    # snd.append(snd1[exam4[0][0]])
    snd.append(st1[exam4[0][0]]+10*fs)  # 稳态踩的时间越长，力越小，固定截取长度减少此误差
    ped_exist.append(ped_index[ii])
ped_list = [np.mean(ped_data[st[i]: snd[i]]) for i in range(len(st))]
# ped_dev = [np.var(ped_data[st[i]: snd[i]]) for i in range(len(st))]
dis_list = [np.mean(dis_data[st[i]: snd[i]]) for i in range(len(st))]
# ped_dev = [np.var(dis_data[st[i]: snd[i]]) for i in range(len(st))]
force_list = [np.mean(force_data[st[i]: snd[i]]) for i in range(len(st))]

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(ped_list, dis_list)
plt.title('ped vs dis')
plt.xlabel('%')
plt.ylabel('mm')
# plt.grid(True)
plt.savefig('ped vs dis', transparent=True)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(ped_list, force_list)
plt.title('ped vs force')
plt.xlabel('%')
plt.ylabel('N')
# plt.grid(True)
plt.savefig('ped vs force', transparent=True)

os.chdir(r'C:\Users\吕惠加\Desktop')
workbook = xlsxwriter.Workbook("ped vs disp & force.xls")
worksheet = workbook.add_worksheet()
worksheet.write_column(0, 0, ['踏板'])
worksheet.write_column(0, 1, ['位移'])
worksheet.write_column(0, 2, ['力'])
worksheet.write_column(1, 0, ped_list)
worksheet.write_column(1, 1, dis_list)
worksheet.write_column(1, 2, force_list)
workbook.close()
