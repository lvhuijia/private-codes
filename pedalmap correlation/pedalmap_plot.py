import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def matrix_to_array(m):
    a = {'p':[], 'x':[], 'y':[]}
    for i in range(0, m.shape[0]):
        row_data = m.ix[i].tolist()
        p = row_data[0]
        row_data = row_data[1:]
        a['p'] += [p]*(m.shape[1]-1)
        a['x'] += m.ix[0].index.tolist()[1:]
        a['y'] += row_data
    return pd.DataFrame(a)


src_path = r'C:\Users\吕惠加\Desktop\pedalmap'
file_list = os.listdir(src_path)

# matrix pedalmap to three columns basic array
# data = pd.read_csv(os.path.join(src_path, 'PTECmap_car.csv'))
# data = matrix_to_array(data)
# data.to_csv(r'C:\Users\吕惠加\Desktop\pedalmap\PTECmap_car1.csv')

target_p = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
color_list = ['magenta', 'indigo', 'blue',
              'steelblue', 'cyan', 'springgreen', 'darkgreen',
              'greenyellow', 'yellow', 'orange', 'tan',
              'chocolate', 'red', 'gray', 'black']

color_map = {}
for i in range(0, len(target_p)):
    color_map[target_p[i]] = color_list[i]

data_dict = {}

for file_name in file_list:
    car_name = file_name.split('.')[0]
    data_dict[car_name] = {}
    if 'csv' not in file_name:
        continue
    data = pd.read_csv(os.path.join(src_path, file_name))
    p_set = np.unique(data['p']).tolist()
    for p in p_set:
        if p in target_p:
            data_dict[car_name][p] = {}
            data_segment = data[data['p'] == p]
            data_drop_flag = ~data_segment['x'].duplicated(keep='last')
            data_drop = data_segment[data_drop_flag]
            data_dict[car_name][p]['x'] = data_drop['x'].tolist()
            data_dict[car_name][p]['y'] = data_drop['y'].tolist()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
lengend_list = []
for car in data_dict:
    for p in target_p:
        if p in data_dict[car]:
            if 'test' in car:
                ax.scatter(data_dict[car][p]['x'], data_dict[car][p]['y'], c=color_map[p], s=2)
            else:
                ax.plot(data_dict[car][p]['x'], data_dict[car][p]['y'], c=color_map[p])
                lengend_list.append(str(p) + '%')
    plt.legend(lengend_list)
# plt.show()
plt.savefig('pedalmap.png')