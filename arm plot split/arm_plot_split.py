import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

src_path = r'C:\Users\吕惠加\Desktop\launch'
file_list = os.listdir(src_path)

target_p = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]

# color_list = ['magenta', 'indigo', 'blue', 'pink',
#               'steelblue', 'cyan', 'springgreen', 'darkgreen',
#               'greenyellow', 'yellow', 'orange', 'tan',
#               'chocolate', 'red', 'gray', 'black']

color_map = {'lamando': 'green', 'GM': 'blue', 'PTECmap': 'red', 'SAIC': 'orange'}

data_dict = {}

for single_p in target_p:
    for file_name in file_list:
        if 'csv' not in file_name:
            continue
        car_name = file_name.split('.')[0]
        data_dict[car_name] = {}
        data = pd.read_csv(os.path.join(src_path, file_name))
        data['p'] = np.array([round(i/5)*5 for i in data['p']])  # 向5取整
        p_set = np.unique(data['p']).tolist()
        for p in p_set:
            if abs(p - single_p) < 1:
                data_dict[car_name][single_p] = {}
                data_segment = data[data['p'] == p]
                data_drop_flag = ~data_segment['x'].duplicated(keep='last')
                data_drop = data_segment[data_drop_flag]
                data_dict[car_name][single_p]['x'] = data_drop['x'].tolist()
                data_dict[car_name][single_p]['y'] = data_drop['y'].tolist()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    lengend_list = []
    for car in data_dict:
        for p in data_dict[car]:
            ax.plot(data_dict[car][single_p]['x'], data_dict[car][single_p]['y'], c=color_map[car])
            lengend_list.append(car + '_' + str(single_p) + '%')
        plt.xlabel('velocity/kph')
        plt.ylabel('acc/g')
        plt.legend(lengend_list)
    # plt.show()
    plt.savefig(src_path + '/' + str(single_p) + '%'+'.png')