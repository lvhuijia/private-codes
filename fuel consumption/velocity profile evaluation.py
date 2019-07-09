import pandas as pd
import matplotlib.pyplot as plt
import os
import xlsxwriter


def velocity_profile_determinate(vel_data, time_step, bin_size):
    max_vel = max(vel_data)
    dis_data = vel_data*time_step/3600  # km
    dis_sum = sum(dis_data)
    label = []
    value = []

    flag = vel_data == 0
    idle_time = int(sum(flag)) * time_step

    for i in range(0, int(max_vel/10) + 1):
        flag = (i*bin_size < vel_data) & (vel_data <= (i+1)*bin_size)
        value.append(sum(flag*dis_data)/dis_sum)
        label.append(str(i*bin_size) + '-' + str((i+1)*bin_size))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.pie(value, labels=label, explode=[0.1]*len(label))
    plt.title('idle_time:' + str(idle_time) + 's')
    plt.show()

    return value, label, idle_time


if __name__ == '__main__':
    time_step = 1  # s
    bin_size = 10
    root_dir = 'D:\个人文档\上汽\工具\车速线\\'
    file_list = os.listdir(root_dir)
    result_dict = {}

    for file_name in file_list:
        if '.csv' in file_name:
            result_dict[file_name] = {}
            data = pd.read_csv(root_dir + file_name)
            vel_data = data['Velocity']  # kph
            result_dict[file_name]['value'], result_dict[file_name]['label'], result_dict[file_name]['idle_time'] = \
                velocity_profile_determinate(vel_data=vel_data, time_step=time_step, bin_size=bin_size)

    workbook = xlsxwriter.Workbook(root_dir + 'results.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write_row(0, 0, ['vel'])
    worksheet.write_column(1, 2 + len(result_dict.keys()), ['idel'])

    counts = 1
    for key in result_dict:
        worksheet.write_column(0, counts, [key])
        worksheet.write_column(1, 0, result_dict[key]['label'])
        worksheet.write_column(1, counts, result_dict[key]['value'])

        worksheet.write_column(0, counts + 2 + len(result_dict.keys()), [key])
        worksheet.write_column(1, counts + 2 + len(result_dict.keys()), [result_dict[key]['idle_time']])
        counts += 1
    workbook.close()


