from scipy import interpolate
import xlrd
import xlwt
from xlutils.copy import copy
import numpy as np
import matplotlib.pyplot as plt


def interpolate1d(x, y, x1):
    inter_function = interpolate.interp1d(x, y, kind='cubic')  # 'slinear', 'quadratic', 'cubic'
    y1 = inter_function(x1)
    x_line = np.arange(min(x), max(x), 0.1)
    y_line = inter_function(x_line)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_line, y_line, c='k')
    ax.scatter(x, y, c='r')
    ax.scatter(x1, y1, c='g')
    ax.legend(['inter_line', 'original', 'inter'])
    plt.show()
    return y1


data = xlrd.open_workbook(r'C:\Users\吕惠加\Desktop\IP34P\ped vs vel.xls')
table = data.sheet_by_index(0)
nrows = table.nrows
ncols = table.ncols
x1 = [table.cell(row, 1).value for row in range(1, nrows)]

data = xlrd.open_workbook(r'C:\Users\吕惠加\Desktop\IP34P\ped vs disp & force.xls')
table = data.sheet_by_index(0)
nrows = table.nrows
ncols = table.ncols
x = [table.cell(row, 0).value for row in range(1, nrows)]
y = [table.cell(row, 1).value for row in range(1, nrows)]  # disp

y1 = interpolate1d(x, y, x1)

y = [table.cell(row, 2).value for row in range(1, nrows)]  # force
y2 = interpolate1d(x, y, x1)

rb = xlrd.open_workbook(r'C:\Users\吕惠加\Desktop\IP34P\ped vs vel.xls')
wb = copy(rb)
ws = wb.get_sheet(0)
ws.write(0, 3, '力')
ws.write(0, 4, '位移')

for i in range(1, len(y1)+1):
    ws.write(i, 3, y2[i-1])
    ws.write(i, 4, y1[i-1])
wb.save(r'C:\Users\吕惠加\Desktop\IP34P\ped vs vel.xls')


