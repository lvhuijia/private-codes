import pandas as pd
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
import os
# 实现稳车速分段提取的显式方法
# flag = [0, 1, 1, 0, 1]
# st2 = []
# snd2 = []
# flag_order = [i for i in range(len(flag))]
# valid_flag_index = [flag_order[i] for i in range(len(flag)) if flag[i] == 1]
# valid_flag_order = [i for i in range(len(valid_flag_index))]
# valid_flag_index_sub_order = [valid_flag_index[i] - valid_flag_order[i] for i in range(len(valid_flag_index))]
# segment_set = set(valid_flag_index_sub_order)
# for i in segment_set:
#     segment_order = [valid_flag_index[j] for j in range(len(valid_flag_index)) if valid_flag_index_sub_order[j] == i]
#     st2.append(min(segment_order))
#     snd2.append(max(segment_order))


def smooth_wsz(a, wsz):
    """

    :param a: 原始数据，NumPy 1-D array containing the data to be smoothed
              必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化
    :param wsz: smoothing window size needs, which must be odd number(if even, -1) ,less than len(a)-1
    :return:
    """
    if isinstance(wsz, int):
        pass
    elif isinstance(wsz, float):
        wsz = int(wsz)
    else:
        wsz = 3

    wsz = min(len(a), wsz)

    if wsz % 2 == 0 and wsz > 2:
        wsz = wsz - 1
    else:  # 无效输入
        wsz = 3

    out0 = np.convolve(a, np.ones(wsz, dtype=int), 'valid') / wsz
    r = np.arange(1, wsz - 1, 2)
    start = np.cumsum(a[:wsz - 1])[::2] / r
    stop = (np.cumsum(a[:-wsz:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


file_path = r'C:\Users\吕惠加\Desktop\IP34-稳态车速 20190305_182404(313_1)_1.csv'
ped_name = 'EPTAccelActuPosHSC1'
# file_path = r'C:\Users\吕惠加\Desktop\IP32-稳态车速 20190306_175921(652_2)_1.csv'
# ped_name = 'AccelActuPosHSC1'
acc_name = 'MSLongAccelGHSC'
veh_name = 'VehSpdAvgNonDrvnHSC1'
fs = 20
veh_spd_index = [10, 20, 40, 59, 79, 98, 108, 118, 127, 137]

st = []
snd = []
veh_spd_exist = []
data = pd.read_csv(file_path, encoding='GB18030')
veh_data = data[veh_name]
pedal_data = data[ped_name]
acc_data = data[acc_name] * 10  # g-->m/s
acc_data = acc_data - np.mean(acc_data)
acc_data = smooth_wsz(acc_data, 5)

for ii in range(0, len(veh_spd_index)):
    exam1 = np.where((abs(veh_data - veh_spd_index[ii]) <= 1) & (pedal_data != 0))
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
    if veh_spd_index[ii] in [10, 20, 40, 60, 80]:
        len1 = 30
    else:
        len1 = 30
    exam4 = np.where(exam > len1 * fs)
    if exam4[0].shape[0] <= 0:
        continue

    st.append(st1[exam4[0][0]])
    snd.append(snd1[exam4[0][0]])
    veh_spd_exist.append(veh_spd_index[ii])

veh_list = [np.mean(veh_data[st[i]: snd[i]]) for i in range(len(st))]
ped_list = [np.mean(pedal_data[st[i]: snd[i]]) for i in range(len(st))]
acc_list = [np.mean(acc_data[st[i]: snd[i]]) for i in range(len(st))]

veh_dev = [np.var(veh_data[st[i]: snd[i]]) for i in range(len(st))]
ped_dev = [np.var(pedal_data[st[i]: snd[i]]) for i in range(len(st))]
acc_dev = [np.var(acc_data[st[i]: snd[i]]) for i in range(len(st))]


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(veh_list, ped_list)
plt.title('veh vs ped')
plt.xlabel('kph')
plt.ylabel('%')
# plt.grid(True)
plt.savefig('veh vs ped', transparent=True)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(veh_list, acc_list)
plt.title('veh vs acc')
plt.xlabel('kph')
plt.ylabel('m/s^2')
# plt.grid(True)
plt.savefig('veh vs acc', transparent=True)

# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.plot(veh_list,veh_dev)
# plt.title('veh vs veh_dev')
# plt.xlabel('kph')
# plt.ylabel('dev')
# # plt.grid(True)
# plt.savefig('veh vs veh_dev', transparent=True)
#
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.plot(veh_list,ped_dev)
# plt.title('veh vs ped_dev')
# plt.xlabel('kph')
# plt.ylabel('dev')
# # plt.grid(True)
# plt.savefig('veh vs ped_dev', transparent=True)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(veh_list,acc_dev)
plt.title('veh vs acc_dev')
plt.xlabel('kph')
plt.ylabel('dev')
# plt.grid(True)
plt.savefig('veh vs acc_dev', transparent=True)

os.chdir(r'C:\Users\吕惠加\Desktop')
workbook = xlsxwriter.Workbook("ped vs vel.xls")
worksheet = workbook.add_worksheet()
worksheet.write_column(0, 0, ['车速'])
worksheet.write_column(0, 1, ['踏板'])
worksheet.write_column(1, 0, veh_list)
worksheet.write_column(1, 1, ped_list)
workbook.close()
