import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
import os
import xlwt
from itertools import groupby
from xlutils.copy import copy
# import xlsxwriter

'字典里面可以放函数对象，通过key去获取并调用指定函数'

'getarr(object, attribute)'
'''获取对象（可以是类、实例等）的属性（可以是方法，函数，属性等）,可以此根据字符串返回函数句柄，从而实现从字符串到函数的路由'''
class class_a:
    def __init__(self):
        self.attribute = 'test'
    def fun_a(self):
        pass
a = class_a()
print(getattr(a,'attribute'))
print(getattr(a,'fun_a'))

'globals(), locals()'
'''以字典的形式返回在全局/局部变量'''
a = 1
'a' in globals()
'a' in locals()
'b' in globals()
'# 结束----------------------------'

'copy.deepcopy(x)'
'''一般赋值中，是把变量指向某内存，当修改该变量时（除非是重新指向其他内存），
会影响到所指向的内存中的变量，赋值时，用deepcopy即可避免内存变量的变动'''
import copy
a = [1,2,3]
b = a
b[0] = 'modified'
print(a)
a1 = [2,3,4]
b1 = []
b1.append(a1)
b1[-1][0] = 'modified'
print(a1)
a2 = [6, 6, 6]
b2 = copy.deepcopy(a2)
b2[0] = 'modified'
print(a2)
'# 结束----------------------------'

'pcolor(xx,yy,zz)'
'''云图，xx,yy为二维数据，一般通过等间隔一维数组用Meshgrid产生，指示平面内每个点的横纵坐标，
zz和xx,yy形式相同，表示每个坐标点上的z值'''
x = numpy.arange(-2, 3, 1)
y = numpy.arange(-2, 3, 1)
xx, yy = numpy.meshgrid(x, y)
zz = xx + yy
qq=plt.pcolor(xx, yy,zz, cmap='jet')
'# 结束----------------------------'

'plt.pie'
'''饼图，explode定义块之间的距离，下面几个参数都是List的形式，等长'''
patches, l_texts, p_texts = plt.pie(sizes,explode=explode,labels=labels,
        colors=colors,autopct='%1.1f%%',shadow=True,startangle=50, labeldistance=1.1)
# labeldistance，文本的位置离远点有多远，1.1指1.1倍半径的位置
# patches, l_texts, p_texts，为了得到饼图的返回值，p_texts饼图内部文本的，l_texts饼图外label的文本
'# 结束----------------------------'

'plt绘图设置'
plt.grid(True)  # 开启x,y轴tick对应的网格线
plt.title('fc: ' + str(4) + '\n' + '66')  # 加标题，可换行
plt.colorbar(fig_handle)  # 加colorbar
fig.set_clim(vmin=0, vmax=150)  # 设置colorbar的两端颜色对应的数值
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文字体
plt.rcParams['axes.unicode_minus'] = False
'# 结束----------------------------'

'__file__'
'''当前运行文件的文件名，和__name__类似'''
os.path.dirname(__file__)  # 返回当前文件名所在的目录路径，目录中用\\分隔
os.path.join(os.path.dirname(__file__), '../')  # 返回当前返回当前文件名所在的目录的上一层路径
'# 结束----------------------------'

'xlwt'
'''写入excel'''
book = xlwt.Workbook(encoding='GB18030', style_compression=0) # 创建workbook
sheet = book.add_sheet('summary', cell_overwrite_ok=True)  # 创建sheet
sheet.write(0, 0, 'a') #在0行0列写入a
sheet.write_merge(0, 1, 2, 3, 'a') # 在0行到1行，2列岛3列的范围内合并单元格，写入a，
book.save('summary.xls') #保存workbook

'xlrd'
'''读excel'''
data = xlrd.open_workbook(r'C:\Users\吕惠加\Desktop\IP34P\ped vs disp & force.xlsx')
table = data.sheet_by_index(0)
nrows = table.nrows
ncols = table.ncols
table.cell(0,0).value # 0行0列
'# 结束----------------------------'

'xlutils.copy'
'''改excel'''
rb = xlrd.open_workbook(r'C:\Users\吕惠加\Desktop\IP34P\ped vs disp & force.xlsx') # 只能读
wb = copy(data)  # 可以写
ws = wb.get_sheet(0)
ws.write(0,0,'a')
wb.save(r'C:\Users\吕惠加\Desktop\IP34P\ped vs disp & force.xlsx') # 复制品覆盖原文件
'# 结束----------------------------'

style = xlwt.XFStyle()  # Create Style
style.alignment.horz = 2  # 字体居中
sheet.write(0, 0, '文本居中', style)
'# 结束----------------------------'

'pd/series.astype(np.float)'
'''转换pd/series的类型，有时候，即使pd/series的每一个子元素都是float，
但是因为整体不是float，就无法做一些整体上的运算，如max,需注意，
series应用此方法后，会replace，dict默认不replace。因此要重新赋值'''
data=[[1,2,3],[4,5,6]]
index=['a','b']#行号
columns=['c','d','e']#列号
df=pd.DataFrame(data,index=index,columns=columns)
df.iloc[0]
df.loc['a']
df.ix['a']
df.ix[0]
if ca_ins.data_assembly[channel][signal].dtype == np.int64:  # convert int64 into float64 since int64 is not feasible for mongo upload (can't convert to bson)
    ca_ins.data_assembly[channel][signal] = ca_ins.data_assembly[channel][signal].astype(np.float)
# int64转化，单个int64元素无法进数据库
'# 结束----------------------------'

'df.iloc[],df.loc[]，df.ix[]'
'''iloc通过行/列号码来索引，loc通过行列名为索引,ix则两种都可以'''
data=[[1,2,3],[4,5,6]]
index=['a','b']#行号
columns=['c','d','e']#列号
df=pd.DataFrame(data,index=index,columns=columns)  #按照columns列表定义的列顺序来生成df
f.iloc[0]
df.loc['a']
df.ix['a']
df.ix[0]
'# 结束----------------------------'

'csv_data = pd.read_csv(self.file_path, encoding='GB18030')'
'''encoding设置编码格式，即使有中文的表头，也不必设置，设置了中文字符集反而会有问题'''
'# 结束----------------------------'

''.'.join(list)'
'''.号把List里面的字符串起来'''
list = ['a', 'b']
'.'.join(list)
'# 结束----------------------------'

'pd.plot(title="XXX", xlim=(0, 10000), ylim=(-30, 100), secondary_y=["VehSpd"])'
'''用dataframe画图，以dataframe index为横坐标，其他列作纵坐标画出曲线系，secondary_y可以设置画在第二个y轴的列'''
pd.set_index("Time")  # 用某一列作为dataframe的index
'# 结束----------------------------'

'function(**dict)----------------------------'
'''把关键字参数用字典的形式包装传给函数'''
'# 结束----------------------------'

'MODEL.objects.filter(**dict)----------------------------'
'''把sql参数用字典的形式包装传给spl语句'''
'# 结束----------------------------'

'reshape(-1,1)----------------------------'
'''ndarraay每个元素撞到一个ndarray中'''
'# 结束----------------------------'

'pd.get_dummies(df)----------------------------'
'''one-hot-encoding'''
data1 = {'a':['a','b','c'], 'b':[2,2,2]}
df1=pd.DataFrame(data1)
df1 = pd.get_dummies(df1)
print(df1.columns)
'# 结束----------------------------'

'df.colname.value_counts()----------------------------'
'''统计某列的所有可能值及其频次'''
data1 = {'a':[1,2,3,3,4], 'b':[2,2,2,2,2]}
df1=pd.DataFrame(data1)
print(df1.a.value_counts())
'# 结束----------------------------'

'np.dot()----------------------------'
'''矩阵相乘'''
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1,2],[3,4],[5,6]])
kk=np.dot(a,b)
'# 结束----------------------------'

'plt.text()----------------------------'
'''按照坐标位置放置文本,x和y需要是浮点数，否则show的时候会有问题'''
x = np.array([1.0, 2.0, 3.0])
y = np.array([1.0, 2.0, 3.0])
text = np.array([1, 2, 3])
for i in range(0, len(x)):
    plt.text(x[i], y[i], str(text[i]))
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.show()
'# 结束----------------------------'

'~True----------------------------'
'''返回为-2，不是False'''
'# 结束----------------------------'

'zip(literal1,literal2) ----------------------------'
'''对于两个可迭代对象，下探一层，按顺序提取再组合成tuple,直到一个Literal下一层的元素遍历完'''
a=[[1,2,3], {'a':1}]  # 下探一层就是a[0]和a[1]，分别是一个List和一个字典
b=[1,2,3]
q=zip(a,b)
qq=list(q)
# 画多个subplot，每个subplot用到的参数不同时的经典做法
fig,axes = plt.subplots(2, 4)
for ax_row, k1 in zip(axes, [10, 100]):
    for ax, k2 in zip(ax_row, [1, 2, 3, 4]):
        ax.plot()
        ax.set_title(str(k1)+str(k2))
plt.show()
axes[0].legend(['1','2'], ncol=2, loc=(.1, 1.1))  #设置legend位置和行列数
'# 结束----------------------------'

'mask = var ==0 ----------------------------'
'''返回ndarray var==0判断的逻辑变量array，可作为mask在dataframe中使用'''
b=np.array([0,1,2,3,4,5])
mask = b ==0
# 找出数组中前n个相同的元素好方法：
a = np.array([0,0,2,1,0,2,1,0,3,3,2,1,3,3,0])
mask = np.zeros(a.shape, dtype=np.bool)
for target in np.unique(a):
    mask[np.where(a == target)[0][:3]] = 1
'# 结束----------------------------'

'# ndarray.ravel()----------------------------'
'''降维，在二维矩阵下，默认按行把矩阵拼成一行'''
x = np.array([[1, 2, 3], [4, 5, 6]])
x.ravel()
'# 结束----------------------------'

'# id(variable)----------------------------'
'''变量在内存中的id，同一id表明同一个内存'''
a=[1,2,3]
id(a)
a=(1,2,3)
id(a)
b=a
id(b)
'# 结束----------------------------'

'# tuple----------------------------'
'''加深内存分配和变量存储原理理解'''
a=(1,[1,2])
a[1] += [3,4]  # 报错，但是仍能让结果生效，报错是tuple模块触发的，但是内存里的数据的确改了
temp = a[1]
temp += [3,4] # 用tuple去索引到存[1,2]的内存块，temp这个指针就不指向tuple了，可以随意更改不报错,
# 相当于存[3,4]的内存块，a(1)和temp都同时指向，只是a(1)是tuple指针，不支持在该指针下修改，temp是普通指针可以
'# 结束----------------------------'

'# 分片----------------------------'
'''a[start:stop:step], start是起数点，stop是终止点（不包含），step为步长，负数表示从右边开始算，此时要求起数点在终止点右边
判断步骤：先根据步长（默认为1）判断步长和方向，再从start开始（如无指明，从最左/最右开始），如果start-->stop不满足步长指定的方向，返回空'''
a = [1,2,3,4,5,6,7,8]
a[1:3] # start=1, stop=3, step=1，stop不包含入内
a[1:-1] # start=1, stop=last, step=1
a[-1:-2:-1] # start=-2, stop=1, step=-1
a[:-2:-1] # start=-2, stop=1, step=-1
a[::-1] # 返回倒序序列
'# 结束----------------------------'

'# ndarray.min(axis=0)----------------------------'
'''沿着ndarray某轴找出最小值，0轴为按列，1轴为按行'''
x = np.array([[1, 2, 3], [4, 5, 6]])
ndarray.min(axis=0)
'# 结束----------------------------'

'# ndarray加减----------------------------'
'''如果加减单个常数，矩阵中逐个加减，如果加减向量，按照向量的形状加减'''
x = np.array([[1, 2, 3], [4, 5, 6]])
x1 = x -1
x2 = x - x.min(axis=0)
'# 结束----------------------------'

'# np.c_[ndarray1, ndarray2]----------------------------'
'''拼接ndarray，当ndarray1 2均为行时，转换为列，再两列并在一起'''
np.c_[np.array([1,2,3]), np.array([4,5,6])]
'# 结束----------------------------'

'# list.index(num)----------------------------'
'''返回list中值为num的第一个Index'''
a=[1,2,3,3]
b=a.index(max(a))
'# 结束----------------------------'

'# np.where(condition, [x, y])----------------------------'
'''返回array，对应位置的元素满足：如果condition=True，返回x对应位置上的元素，否则返回y对应位置上的元素,若不指明x和y，则返回condition!=False/0的顺次号
    If `x` and `y` are given and input arrays are 1-D, `where` is
    equivalent to:: [xv if c else yv for (c,xv,yv) in zip(condition,x,y)]'''
a=np.where([False,True,False])
np.where([True, False,True, True],[[1, 2, 3, 4]],[[9, 8, 7, 6]])
'# 结束----------------------------'

'# enumerate(iterable)----------------------------'
'''产生一个序列对'''
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
'# 结束----------------------------'

'# groupby(iterable[, keyfunction])----------------------------'
'''from itertools import groupby 根据keyfunction定义的内容来对iterable内容分组'''
groupby(enumerate(np.array(np.where(flag))),lambda x: x[1] - x[0])  # 用于给连续点系列分组的函数，如flag=[0,1,1,0,1]时，np.where(flag)返回1，2，4
# enumerate(np.array(np.where(flag)))产生# (0,1),(1,2),(2,4),lambda函数定义每个tuple中后减前，以减出来的数作为判据去分组，即分成差为1和差为2两组
# 可以用for i,j in groupby(...)循环引用，其中i为第i组，j为对应组的grouper object,通过Next可以看到组里面的tuple元素，一个tuple中，第一个为序号，第二个为组内的具体元素list
'# 结束----------------------------'

'# lambda函数----------------------------'
'''匿名函数,语法为：lambda argument_list: expression
一般功能简单：单行expression决定了不可能完成复杂的逻辑。由于其实现的功能一目了然，不需要专门的名字来说明。
部分Python内置函数接收函数作为参数，如map，groupby等'''
f=lambda x:x+1  # x为输入参数，x+1为输出结果
f(0) # 输入0的返回值
foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
filter(lambda x: x % 3 == 0, foo).__next__()
# [18, 9, 24, 12, 27]
map(lambda x: x * 2 + 10, foo).__next__()
# [14, 46, 28, 54, 44, 58, 26, 34, 64]
reduce(lambda x, y: x + y, foo).__next__()
# 139
lambda *args: sum(args) # 输入是任意个数的参数，输出是它们的和(隐性要求是输入参数必须能够进行加法运算)
lambda **kwargs: 1 # 输入是任意键值对参数，输出是1
time.sleep=lambda x:None # 把内置的sleep函数功能屏蔽掉
# return lambda x, y: x+y # 返回一个加法函数。lambda函数实际上是定义在某个函数内部的函数，称之为嵌套函数，或者内部函数。
# 对应的，将包含嵌套函数的函数称之为外部函数。内部函数能够访问外部函数的局部变量，这个特性是闭包(Closure)编程的基础。
lambda: {'name': [], 'feature': []}  # 没有输入，统一返回固定字典的匿名函数
'# 结束----------------------------'

'# pandas.DataFrame(data, columns=[a, b, c])----------------------------'
'''由data创建dataframe，如果data是list（用List嵌套来声明多列）,则列名从0开始，也可以用columns，如果data是dict，则列名为key，内容为items，
列按照dict的key首字母排列，也可由columns来指明顺序'''
data1 = {'a':[1,2,3,3,4], 'b':[2,2,2,2,2]}
df1=pd.DataFrame(data1)
df1['a']
df1.a
df1.a[:,np.newaxis]  # 提取名为'a'的一列，重新生成一个ndarray
df1.duplicated(keep='last')  # 着眼整个df，如果某行和之后出现过的完全一致，该行号返回True，否则返回False
df1['a'].duplicated(keep='last')  # 着眼df的某一列，如果某行和之后出现过的完全一致，该行号返回True，否则返回False
df1[df1['a'] > 2001]  # 逐个比较得到一个包含逻辑量的series,可以直接作用在df上，从而实现提取
data1.notnull()*1  # 非null置1，null置0，null包括np.nan和None
data2 = [[1,2,3], [None,2,2]]
df2=pd.DataFrame(data2)
df2[0].dropna()  # 如果df是一列，丢弃Null的，如果是矩阵，丢弃含有油Null的行
csv_data.index = [time_start + timedelta(seconds=time) for time in csv_data['Time (abs)']]
# 给df绑定一个index，一般是时间戳，各列提取出来的时候都带时间信息。
# df下包含的每列是Series数据类型（含有数据索引），如无指明，数据索引为0，1...，数据索引也可以通过df.index来指定
'# 结束----------------------------'

'# 产生时间序列并和数据组装成时间索引的series并重采样----------------------------'
''''''
index = pd.date_range('1/1/2000', periods=9, freq='T')  # T也可以是D，M,Y等其他，形成等间隔的datetimeindex，T代表按1分钟等间隔，3T代表3分钟，1U代表1微秒
series = pd.Series(range(9), index=index)
series.resample('3T').sum()  # series.resample('3T')产生一个resampler，.sum（）把重采样时间间隔之间覆盖区间各值累加起来
series.resample('3T', label='right').sum()  # 重采样时间间隔之间的时间区间，时间Label统一取右边时间，默认为左边
'# 结束----------------------------'

'pandas重采样：df.resample(r_r).ffill()'
'''r_r是str类型，值为str(int(1e9/重采样频率)+'N'(1s=1e9ns，意思是用字符串的形式表达每隔多少ns重采样一个点)
df要求index为 datetime-like index (DatetimeIndex,PeriodIndex, or TimedeltaIndex)，可以人为构造时间戳，并赋值给df.index
ffill()是用来填充NaN值的，ffill表示front fill，NaN用前面的邻近有效值来填充，bfill()表示向后填充，NaN用后面的邻近有效值来填充
'''

'# parse(timestr, parserinfo=None, **kwargs)----------------------------'
'''由dateutil.parser导入，用于解析字符串形式的日期信息,kwargs包含yearfirst,如果是True，即7/25/2018第一部分7是年份，否则最后一部分2018是年份，默认是False
dayfirst=True代表第一个是日，否则第一个是月，默认是False，逻辑是先根据yearfirst确定年的位置，如果年是第一，再根据dayfirst确定是YMD还是YDM,另外，会根据YDM
的取值范围来修正结果，不会机械式套用yearfirst和dayfirst，只有在有多种个能的时候才会参考两个first'''
parse('7/25/2018-6:39:24 pm')
parse('7/8/10-6:39:24 pm', yearfirst=True, dayfirst=False)
'# 结束----------------------------'

'# sort_values----------------------------'
'''对dataframe的某一（多）列进行排序，多列排序有主次之分，ascending=True决定升序降序，inplace=True表示排序后替代原dataframe，否则不影响原df'''
csv_data.sort_values(by=['Time (abs)','acc'], inplace=True)
'# 结束----------------------------'

'# xlsxwriter----------------------------'
'''支持对xlsx格式文件的输出，行列数限制低，可带格式写入，生成excel'''
workbook = xlsxwriter.Workbook("chart_line.xlsx")
worksheet = workbook.add_worksheet()
process_assembly_name = [['asad']]
worksheet.write_row(0, 0, process_assembly_name[0])
# 0,0表示起点的行和列，也可以用‘A1’来等效代替，第三个data中会向下解构一层，如果process_assembly_name[0]是字符串，会把每个字符分开写入单元格中，如果是list，全部写到一个格子中
worksheet.write_column(1, 1, process_assembly_name[0])
'# 结束----------------------------'

'# pd.concat([df1,df2]...)----------------------------'
'''dataframe合并，可以沿着列方向或者行方向拼接，从而拓展行或者列，且能处理拥有不同列名的df合并，可以设置交集外的列用NAN表示或者舍去，还有很多设置
第一个参数必须是可迭代的pd object，因此要用list把series/pd等元素括起来'''

'# 结束----------------------------'

'# 参数收集原理----------------------------'
'''定义时和位置参数一样，只是可以提供默认值，如果没有用关键字调用，
仍按位置参数赋值，只是如果实参不够时用默认值，且不能重复赋值'''
def collect_para(x, y, z=3, *pospara, **keypara):
    print(x, y, z)
    print(pospara)
    print(keypara)


collect_para(1, 2, z=5, foo=1, bar=2)
collect_para(1, 2, 4, z=5, foo=1, bar=2)
'# 结束----------------------------'


'# np.linspace(start, end, num)----------------------------'
'''endpoint默认为True，表示10也包括进去'''
x = np.linspace(0, 10, num=11, endpoint=False)
'# 结束----------------------------'


'# np.arange(start, end, step)----------------------------'
'''不含endpoint,ndarray中a[1:3]同理'''
x = np.arange(0, 10, 11)
'# 结束----------------------------'


'# docstring----------------------------'
'''在模块、包、类、函数特定位置用3对单/双引号括起来写入，
用help函数查询时显示该部分说明'''
'# 结束----------------------------'

'# string.strip(string1)----------------------------'
'''去除字符串string的首尾所包含特定的字符string1，首尾的概念是，源字符串第一个字符，如果去掉后，那么第二个字符成为首字符，如此下去，直到某字符成为
首字符时，不用删去，则结束；lstrip只删除首，rstrip只删除末'''
a=" \rzha ng\n\t "
b=a.strip() # 如不传入参数，去掉首尾的空白字符，包括空格，换行等制表符（中间的空格保留）
print(b)
a="rrbbrrddrr"
b=a.strip("r")
a="aabcacb1111acbba"
print(a.strip("abc"))  # 传入abc，则a/b/c分开，同时作用于字符串，具体步骤：第一个a判断，满足条件，删除，第二个a判断，满足删除。。。直到1不满足
'# 结束----------------------------'

'# np.array()----------------------------'
'''用list包list的形式传参，生成shape=(M,D)的array（M*D）
，其中D为小list的len,M为大list的，因而排布是从上到下堆叠'''
c = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
'# 结束----------------------------'

'# np.vstack((arrayA,arrayB))----------------------------'
'''把A,B两个array从上到下堆叠，因此要求a和b具有相同的列数；hstack则是横向堆叠拼接;dstack是打破array的元素，在元素层面拼接
dstack常用于由灰度图通过RGB颜色方案重构黑白图：np.dstack((gray,gray,gray))*255'''
c = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
d = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
e = np.vstack((c,d))
f = np.hstack((c,d))
c = np.array([[1], [1], [1], [1]])
d = np.array([[1], [1], [1], [1]])
g = np.dstack((c,d))
'# 结束----------------------------'

'# np.flipud(array)----------------------------'
'''array上下翻转（以中间行为对称轴对折翻转），npfliplr()则是左右翻转'''
c = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
d = np.fliplr(c)
e= np.fliplr(d)
'# 结束----------------------------'

'# XX, YY = np.meshgrid(x, y)----------------------------'
'''用N个一维向量创建返回N个N维网格，每个一维向量返回对应的
N维网格中，只有沿着该一维向量方向上数值改变，其他方向都是复制，
和坐标系（第一个一位数组当作X坐标处理，返回数组中沿着X方向即打横改变）
、ij下标规定类似（第一个一位数组当作i处理，返回数组中沿着i方向即打竖改变）'''

'# 结束----------------------------'


'# mgrid = np.mgrid(start:end:step, start:end:step)----------------------------'
'''与上面类似，调用时输入范围，指定维数，j表示包含终点'''
np.mgrid[0:1:100j, 0:1:200j]
'# 结束----------------------------'


'#  np.random.rand(row, col)----------------------------'
'''返回row行col列0-1随机数'''
np.random.rand(1000, 2)
'# 结束----------------------------'


'''#  scipy.interpolate.griddata(points, values, xi, method='linear', fill_value=nan, rescale=False)----------------------------'''
'''用非规则的data插值,points为m行两列array（每一行仍是一个array），每一行存插值点x,y，
values为m行一列array，每行存放与points对应的z（每一行为List），xi的shape为（M，D），
即从外部看来，第一次拆分需要可以拆成D维，tuple和ndarray都可以做到，第二步拆分就是从上
一步得到的D个子元素中，逐一拎出单个数字，如果子元素为matrix，就得到matrix形式的坐标点，
为vector则得到vector形式的坐标点'''
# shape是array的属性之一，shape(n,D)表示len=n, Dim=D, 直观显示为n行，D列，
# 因而points的shape为（n,D)，value为（n,）一维省略1
'# 结束----------------------------'


'#  样条曲线----------------------------'
'''样条曲线不仅通过各有序型值点，并且在各型值点处的一阶和二阶导数连续，
也即该曲线具有连续的、曲率变化均匀的特点'''
'# 结束----------------------------'


'#  Axes3D.plot_surface(X,Y,Z,*arg, **kwarg)----------------------------'
'''X,Y,Z均为2D array默认纯色，用关键字参数cmap可指定color map，rstride和cstride用来指定步幅，
默认10,10，例如输入1k*1k数据，共产生100*100个网格去上色，X，Y中依次各取数组合成坐标，
Z中读取数据的方向为从上到下，从左到右，即按列读取'''
x = np.linspace(-1, 2, 300)
y = np.linspace(-1, 1, 300)
xx, yy = np.meshgrid(x, y)
zz = xx + yy

fig = plt.figure()  # 生成画布
ax0 = Axes3D(fig)   # 在画布上生成轴
surf = ax0.plot_surface(xx, yy, zz, rstride=10, cstride=10,
                       cmap=cm.coolwarm, linewidth=0.5)  # 在轴上画图


ax = plt.subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(xx, yy, zz, rstride=10, cstride=10,
                       cmap=cm.gist_rainbow_r, linewidth=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.colorbar(surf, shrink=0.5, aspect=5)

ax1 = plt.subplot(1, 2, 2, projection='3d')
surf = ax1.plot_surface(x, y, zz, rstride=10, cstride=10,
                       cmap=cm.coolwarm, linewidth=0.5)  # x,y中各取一个元素作为(x,y)，对应的zz array中取长度为300的array，因而画出来是多值函数
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.colorbar(surf, shrink=0.5, aspect=5)
'# 结束----------------------------'

'#  Axes3D.plot_surface(X,Y,Z,*arg, **kwarg)----------------------------'
'''画二维图'''
x = np.linspace(-1, 2, 300)
y = np.linspace(-1, 1, 300)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, color='green', linestyle='dashed', marker='*', markerfacecolor='blue',
         markersize=8)


'#  dict(key1=value1,key2=value2,)----------------------------'
'''定义字典两种方法等效，Pop函数根据key返回value并删除字典内相关内容，并可设置默认值，
当key不存在时，返回默认值'''
dic1 = dict(name1=1, name2=2)  # 这里name是关键字参数传入，不用引号
dic2 = {'name1': 1, 'name2': 2}  # 这里name是字符串，要引号
dic1 == dic2
dic1.pop('name1')
dic1.pop('name3', 'not exist')
'# 结束----------------------------'

'#  dict.update(dict)----------------------------'
'''用一个字典更新另外一个字典，如果没有的key会被添加，若key存在，则value被替代，注意！！！当字典update后，之前这个字典被赋到其他地方时，这些地方的值会随之改变'''
dic1 = {'s': 1}
list1  = []
list1.append(dic1)
dic1.update({'s': 2})

'# 结束----------------------------'


'#  dict(key1=value1,key2=value2,)----------------------------'
'''定义字典两种方法等效，Pop函数根据key返回value并删除字典内相关内容，并可设置默认值，
当key不存在时，返回默认值'''
dic1 = dict(name1=1, name2=2)  # 这里name是关键字参数传入，不用引号
dic2 = {'name1': 1, 'name2': 2}  # 这里name是字符串，要引号
dic1 == dic2
dic1.pop('name1')
dic1.pop('name3', 'not exist')
'# 结束----------------------------'


'#  类内函数定义----------------------------'
'''不加前缀：普通函数，默认传入实例self；@static，静态函数，默认不传参，@classmethod，默认传入类名cls'''
class TClassStatic(object):
    obj_num = 0  # 定义Class下面,init之前，为定义类时赋给类的一个属性，用类名.属性名可以引用，实例化时跳过此步，直接从init开始
    def __init__(self, data):
        self.data = data  # self/类名表示对实例或类属性进行定义
        TClassStatic.obj_num += 1

    def printself(self):  # 传入实例，哪个实例调用此函数就传哪个实例
        print("self.data: ", self.data)

    @staticmethod
    def smethod():  # 无参数
        print("the number of obj is : ", TClassStatic.obj_num)

    @classmethod
    def cmethod(cls):  # 传入类，哪个类调用此函数就传哪个类
        print("cmethod : ", cls.obj_num)
        cls.smethod()

def main():
    objA = TClassStatic(10)
    objB = TClassStatic(12)  # 实例化B时，objnum作为类属性，会同步在objA和objB里面更新
    objB.printself()
    objA.smethod()  # 传如objA
    objB.cmethod()  # 传如objB所属类：TClassStatic
    print("------------------------------")
    TClassStatic.smethod()
    TClassStatic.cmethod()
    TClassStatic.printself()  # 没有实例，不能调用
'# 结束----------------------------'


'#  np.convolve(a, v, mode)----------------------------'
'''求一维离散数组a*v的卷积，卷积定义为对t积分（求和）a(t)*v(-tao+t)的乘积，
理解成将v关于y轴对称，再加偏移量t，t从负无穷到正无穷变化，t每取一个值，
v函数变换后有一个定义域，如果定义域与a相交，则乘积之和不为0，
定义为t处卷积的函数值。mode可选择full（定义域相交就算，返回长度为N+M-1），
same(从相交开始算，算到出现不相交位置，返回长度为max（N,M)，valid(定义域
完全包含才算，返回长度为max(M, N) - min(M, N) + 1)，把v定义成权函数，就可
以实现对a滤波/平滑，注：定义域边界重叠点也算'''
'# 结束----------------------------'


'#  pd.DataFrame(data, index=rowsname, columns=columnsname)----------------------------'
'''用data，rowsname作为行名和columnsname作为列名创建数据表格，
即用数据矩阵和字符创建形如实验输出格式的表格数据,行名列名默认从0开始'''
data=[[1,2,3],[4,5,6]]
index=['a','b']#行号
df=pd.DataFrame(data,index=index)#生成一个数据框
df.loc['a',:]
'# 结束----------------------------'


'''#  if __name__ == '__main__':----------------------------'''
'''当该模块（该py文件）直接运行时，if内语句运行，当被调用时(__name__为本名，
且包含层级从属关系)，不运行，被调用时，所有
最上层的语句都会被执行（即无缩进部分）
在哪个py点运行，该py就会作为main函数，在此py内执行语句时，__name__一直为__main__'''
'# 结束----------------------------'


'#  列表推导式:----------------------------'
'''结合循环和判断来建立逻辑数组，逻辑数组与原始数据相乘，达到某些值置零的效果'''
b = [1,0,1]
a = [1 if x ==0 else 0 for x in b ]  # 判断，必须完备，再循环，即每次循环需要写入
a = [x+y for x in b for y in b]  # 各元素组合相加，得到3*3
b = [b[0]+i for i in b]  # 不会动态刷新b[0]，整个list计算完毕后再赋值给b
b = [x for x in b if x > 0]  # 循环，判断，通过就写入，即每次循环不一定写入
'# 结束----------------------------'


'#  表中存数对:----------------------------'
''''''
a=np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')])
'# 结束----------------------------'


'#  字符串操作:split----------------------------'
'''按照特定字符给字符串分段,返回一个分段切割后的字符串list'''
a='a/b/c.d'
a.split('/')[2]  #list中第三各，即第三段，'c.d'
rawdata_name = a.split("/")[-1].split(".")[0]  #获取路径中包含的文件名
'# 结束----------------------------'


'#  列表操作----------------------------'
a=[1,2,3,4]
a.insert(0,2)  # 在原来0位置左侧插入
a.insert(-1,998)  # 在原来最后的位置左侧插入
a.split('/')[2]  #list中第三各，即第三段，'c.d'
rawdata_name = a.split("/")[-1].split(".")[0]  #获取路径中包含的文件名
'# 结束----------------------------'

'#  字符串操作:replace----------------------------'
'''用特定字符取代特定字符'''
a='a/b/c.d'
a.replace('.','/')  #/代替.
'# 结束----------------------------'


'#  字符串操作:join----------------------------'
'''用特定字符连接List中的所有子字符串'''
a='a/b/c.d'
a= ' '.join(a.split())  #split空参数默认把字符间的空格去掉来分割
'# 结束----------------------------'


'#  逻辑表达式：罗技运算符的左右元素用括号括起来！！----------------------------'
'''否则运算法则会乱'''
a=1>0&(2<1) # 先算括号，0.再算逻辑和，0，最后算大于，1
a=(1>0)&(2<1) # 先算括号，1,0.再算逻辑和，0
'# 结束----------------------------'


'#  os操作----------------------------'
os.getcwd()  # 当前工作目录
os.chdir()  # 切换工作目录
os.listdir('filepath')  # 显示目录下的素有文件
os.path.join(path, path)  # 把路径和文件名组装成该文件的路径
os.path.isfile(path)  # 传入绝对路径，判断是否是文件
os.path.isdir(path)  # 传入绝对路径，判断是否是文件夹
'# 结束----------------------------'

'#  os.walk(origin_path)----------------------------'
'''从origin_path开始往下遍历目录树，返回每一个文件夹的路径，里面的文件夹名，文件名，遍历顺序为先纵向后横向'''
for dirpath, dirnames, filenames in os.walk(origin_path):
    pass
# root
#   |
# folder1 folder2 file1 file2
#   |
# folder3 folder4 file3 file4
# 第一个循环：dirpath= root的路径 dirnames=[folder1 folder2], filenames = [file1 file2]
# 第二个循环：dirpath= folder1的路径 dirnames=[folder3 folder4], filenames = [file3 file4]
'# 结束----------------------------'

'#  目录设置----------------------------'
'''默认工作目录为project目录，可通过file-new新建project来改变，file-setting-tool-terminal目录随project设置而改变，直接设置貌似无效'''
'# 结束----------------------------'

'#  带名字元组----------------------------'
'''元组内容可命名，索引时可以通过数字下表，也可以通过名字像变量属性一样索引'''
from collections import namedtuple, defaultdict
AAA = namedtuple('DATA_INFO', ['FullPath', 'BasePath', 'Mode', 'GK', 'SubGK'])  # AAA类似一个类，接受顺序输入或者kwarg,生成的属性可以用两种方式索引
qq = AAA('3','4','5', GK='1',SubGK='2')
qq[0]  # 索引输入顺序第一的属性
qq.FullPath  # 按名字索引，和输入顺序无关
'# 结束----------------------------'

'#  getattr(object, name, default=None)----------------------------'
'''获得object的name属性'''
getattr(x, 'y', 'nothing')  # 返回x类的y属性
getattr(x, 'y', 'nothing')()  # 运行x类的y方法
'# 结束----------------------------'

'#  连续赋值----------------------------'
'''连续赋值会使几个变量指向同一个内存区，修改的时候联动修改,如果是重新给变量赋值，则该变量指向另一个内存区，剥离出来'''
a = b = c = []
a.append(2)
b.append(3)
a=1
'# 结束----------------------------'

'#  and or...----------------------------'
'''两边需是逻辑值(或数字），如传入list也可计算不报错，但并不是按位计算返回一个逻辑值的List,
而是or总是返回左边的list，and总是返回右边的List.对于数字.
旧理解：or从左到右，遇到非0就返回并退出，直到最右没退出(即全是0）就返回最右/最左，
and从右到左遇到0就返回退出，直到最左没退出就返回最右。如果左右是np，则按位运算并返回ndarray'''
a=[1,1,1]
b=[0,1,1]
c=[0,0,0]
a or b
b or a
c or a
a or c
a and b
b and a
a and c
a=np.array(a)
b=np.array(b)
a & b
b & a
'# 结束----------------------------'

'#  & |...----------------------------'
'''两边需是逻辑值(或数字），左右是np，则按位运算并返回ndarray'''
a=[1,1,1]
b=[0,1,1]
a=np.array(a)
b=np.array(b)
a & b
b & a
'# 结束----------------------------'

'describe()----------------------------'
'''计算返回计数，平均，最大最小，中位数，75%以上分位，25%以下分位等参数，用来画箱型图'''
p = pd.DataFrame(i_pedal, columns=['i_pedal']).describe()
'# 结束----------------------------'

'defaultdict(type/function)----------------------------'
'''生成带有默认返回的dict，和dict不是同一类型，当索引的key不存在时，返回定义的默认值'''
from collections import defaultdict
a = defaultdict(lambda: 0)  # 默认配0
b = defaultdict(list)  # 默认配空Lish
'# 结束----------------------------'

'画图弹窗----------------------------'
'''不知道为什么会弹窗'''
# from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 100)
fig = plt.figure()
fig.add_subplot(221).plot(x, x)
fig.add_subplot(221).plot(x, -x)

fig.add_subplot(221).legend(['a','b'])
ax2 = fig.add_subplot(221)
ax2.plot(x, -x)
ax3 = fig.add_subplot(223)
ax3.plot(x, x ** 2)
ax4 = fig.add_subplot(224)
ax4.plot(x, np.log(x))

fig2 = plt.figure()
fig2.add_axes([0.15, 0.1, 0.75, 0.8]).plot(x, x)
fig2.add_axes([0.15, 0.1, 0.75, 0.8]).plot(x, -x)
fig2.add_axes([0.15, 0.1, 0.75, 0.8]).set_xlim(0,100)
plt.show()
'# 结束----------------------------'

'string.format(string1)----------------------------'
'''用string1代替string中{}内的内容，和string %s (string1)代替string中%内容同理，
但是format中{}和string1不必顺次对应，可多次引用，{0}对应string1中的第一个，以此类推，
另外可以对数字输出格式化'''
print("{:.2f}".format(3.1415926))  # 数字输出格式化
print('{}，{}'.format('test','6'))  # 顺次对应
print('%s，%s' % ('test','6'))  # 顺次对应
print('{1}，{0}，{1}'.format('test','6'))  # 非顺次对应，多次引用
print('{{}}，{},{}'.format('test','6'))  # 括号转义
'# 结束----------------------------'

'pandas.merge((left, right, how=\'inner\', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=True, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)'
''''''
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                       'A': ['A0', 'A1', 'A2', 'A3'],
                       'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']})
result = pd.merge(left, right, on='key')  # 以key列为键拼接，两个df中，key列相同的行拼接，如果左右df的key列中有不同的元素，则舍去

'# 结束----------------------------'

'多轴柱状图----------------------------'
'''多轴legend原理通用,通过bar width和x=[0,1,2...]来对柱状图定位，再把x的标签修改成想要的值'''
bar_list = []
legend_list = []
fig = plt.figure()
x=[0, 1, 2, 3]
bar_width = 0.3
ax1 = fig.add_subplot(1,1,1)
ax2 = ax1.twinx()
bar1 = ax1.bar([i for i in x],[1,1,1,1],width=bar_width, color='r', label='1')
bar2 = ax2.bar([i+bar_width for i in x],[1,2,3,4],width=bar_width, color='g', label='2')
bar3 = ax1.bar([i-bar_width for i in x],[2,3,4,5],width=bar_width, color='black', label='3')
bar_list.append(bar1)
bar_list.append(bar2)
bar_list.append(bar3)
legend_list.append('fuel consumption(L/100km)')
legend_list.append('electric consumption(L/100km)')
legend_list.append('net fuel consumption(Kw/100km)')
ax1.legend(bar_list, legend_list,loc='upper left')
ax1.set_xticks([0,1,2,3])
ax1.set_xticklabels(['a','b','c','d'])
ax1.set_ylim([-2, 16])  # 设置lim和ticks来保证坐标轴对齐，只设置ticks，整个周可能会有偏移，导致对不齐
ax2.set_ylim([-10, 35])
ax1.set_yticks([-2 + i * 2 for i in range(0, 10)])
ax2.set_yticks([-10 + i * 5 for i in range(0, 10)])
ax1.set_xlabel('velocity(kph)')
ax1.set_ylabel('fuel consumption')
ax2.set_ylabel('electric consumption')
fig.show()
'# 结束----------------------------'

'装饰器示例----------------------------'
'''缓存'''
# 装饰器增加缓存功能
def cache(instance):
    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            joint_args = ','.join((str(x) for x in args))
            joint_kwargs = ','.join('{}={}'.format(k, v) for k, v in sorted(kwargs.items()))
            key = '{}::{}::{}'.format(func.__name__,joint_args, joint_kwargs)
            result = instance.get(key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            instance.set(key, result)
            return result
        return inner_wrapper
    return wrapper


# 创建字典构造函数，用户缓存K/V键值对
class DictCache:
    def __init__(self):
        self.cache = dict()

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value

# 创建缓存对象
cache_instance = DictCache()

# Python语法调用装饰器
@cache(cache_instance)
def long_time_func(x):
    return x

# 调用装饰过函数
long_time_func(3)
'# 结束----------------------------'

'yield+循环应用示例----------------------------'
'''循环yield，直到generator耗尽'''
while True:
    try:
        subgk_info = file_router.__next__()
    except StopIteration:
        break
'# 结束----------------------------'

'pandas综合应用案例'
'''有一个记录换挡点的df，需要扩展为记录了所有时刻所在档位的df'''
df = pd.DataFrame({'time':[1,5,9], 'gear': [1,2,3]})  # 换挡点
df_time = pd.DataFrame({'time': range(0, 11)})  # 构造所有时刻序列
df_ful = pd.merge(df, df_time, how='outer', on='time')  # 合并df，on=time表示以time列为基准合并，
# how=outer表示用两个df的time列元素并集合并（inner表示交集），没有的用NAN填充，因为df_time是完整的时间序列，所以outer可以改为right
df_ful.sort_values(by='time', inplace=True)  # 按时间列排序，形如df.xx的操作，注意inplace设置为True，否则要把结果赋值给新变量
df_ful.ffill(inplace=True)  # 填充NAN向前取值
df_ful.bfill(inplace=True)  # 把第一个NAN填充掉
'# 结束----------------------------'

'字符串转换为数字'
'''浮点数和整型数转换方式不同'''
str1 = '1.0'
str2 = '1'
num1 = float(str1) # 不能用int
num2 = int(str2) # 可以用float
'# 结束----------------------------'

'读txt'
'''逐行读，注意换行符在文档中是不可见的，但是能读进来'''
with open(file_path, 'r') as f:
    while True:
        line = f.readline()
        if not line: # 非空行,注意可能是中间的空行，也可能是最后一行
            break
        else:
            pass  # 对每一行信息提取操作等
'# 结束----------------------------'