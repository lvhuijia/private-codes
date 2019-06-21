import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import xlwt
import xlsxwriter
import timeit
import pickle

# initialize calculation parameters
dbc_dic = {'A18SOV120': {'pedal': 'AccelActuPosHSC1', 'velocity': 'LSSpeed', 'brake': 'BrkPdlDrvrAppdPrs_H1HSC1', 'speed': 'EnSpdHSC1', 'torque': 'EnToqActuExtdRngHSC1', 'gear': 'TrGearHSC1'},
           'A18SOV121': {'pedal': 'AccelActuPosHSC1', 'velocity': 'LSSpeed', 'brake': 'BrkPdlDrvrAppdPrs_H1HSC1', 'speed': 'EnSpdHSC1', 'torque': 'EnToqActuExtdRngHSC1', 'gear': 'TrEstdGear_TCMHSC1'},
           '卡罗拉': {'pedal': 'corolla_AccPedPos', 'velocity': 'LSSpeed', 'brake': 'corolla_BrkLightSwitch', 'speed': 'corolla_EngineSpeed', 'torque': 'corolla_DrvDesTorqRaw', 'gear': 'corolla_ActGearRatio'},
           '途昂': {'pedal': 'MQB_AccPedPos', 'velocity': 'Speed', 'brake': 'MQB_BrkPedSwitch', 'speed': 'MQB_EngineSpeed', 'velocity_backup': 'MQB_VehicleSpeed', 'torque': 'MQB_EngineTorqueWInt',  'gear': 'MQB_ActualGearPos'},
           '汉兰达': {'pedal': 'corolla_AccPedPos', 'velocity': 'LSSpeed', 'brake': 'corolla_BrkLightSwitch', 'speed': 'corolla_EngineSpeed', 'velocity_backup': 'corolla_VehicleSpeed1s', 'torque': 'corolla_ActToqClth', 'gear': 'highlander_Gear'},
           'AS23': {'pedal': 'AccelActuPosHSC1', 'velocity': 'LSSpeed', 'brake': 'PtBrkPdlDscrtInptStsHSC1', 'speed': 'EnSpdHSC1', 'velocity_backup': 'VehSpdAvgNonDrvn_h1HSC1', 'torque': 'EnToqActuExtdRngHSC1', 'gear': 'TrEstdGearHSC1'}
           }
main_ratio_dic = {'A18SOV120': 5.7, 'A18SOV121': 1.0, '卡罗拉': 1.0, '途昂': 1.0, '汉兰达': 1.0, 'AS23': 1.0}
fre = 100
hold_time_min = 6  # used for data segment filtering
velocity_start_max = 2  # used for segment filtering
velocity_end_min = 100/3.6  # used for calculating 0-100 time
velocity_start_min = 0.5/3.6  # used for calculating 0-100 time
lead_time = 1  # used for cutting more data for 0-100

# list for summary
order = []
speed_at_start = []
velocity_at_start = []
velocity_at_end = []
time_for_acc = []

pipe_inlet_at_start = []
turbine_outlet_at_start = []
cooling_outlet_at_start = []
cooling_to_valve_at_start = []
valve_inlet_at_start = []
outside_at_start = []

pipe_inlet_at_end = []
turbine_outlet_at_end = []
cooling_outlet_at_end = []
cooling_to_valve_at_end = []
valve_inlet_at_end = []
outside_at_end = []

vehicle_number = []
temperature_target = []
frequency_cal = []
frequency_int = []

# list for procedure data
time_data_section = []
torque_data_section = []
speed_data_section = []
velocity_data_section = []
gear_data_section = []
count = 0

file_dir = r'C:\Users\吕惠加\Desktop\动力衰减'
file_list = os.listdir(file_dir)
xls_list = [file for file in file_list if '.csv' in file]
# initialize calculation parameters ends

for file_number in range(0, len(xls_list)):
    xls_name = xls_list[file_number]
    print(xls_name + ' begins calculating')
    info = xls_name.split('_')
    vehicle_info = info[0]
    temperature_info = info[1]
    test_info = info[2]
    xls_path = os.path.join(file_dir, xls_name)
    data_ful = pd.read_csv(xls_path, header=14, skiprows=[15, 16], skip_blank_lines=False, encoding='GB18030')
    if '序号' not in data_ful.columns:
        data_ful = pd.read_csv(xls_path, header=0, skip_blank_lines=False, encoding='GB18030')
    pedal_dbc = dbc_dic[vehicle_info]['pedal']
    velocity_dbc = dbc_dic[vehicle_info]['velocity']
    speed_dbc = dbc_dic[vehicle_info]['speed']
    brake_dbc = dbc_dic[vehicle_info]['brake']
    torque_dbc = dbc_dic[vehicle_info]['torque']
    gear_dbc = dbc_dic[vehicle_info]['gear']
    pedal_data = data_ful[pedal_dbc].tolist()
    velocity_data = data_ful[velocity_dbc].tolist()
    speed_data = data_ful[speed_dbc].tolist()
    brake_data = data_ful[brake_dbc].tolist()
    torque_data = data_ful[torque_dbc].tolist()
    torque_data = [number * 1.0 for number in torque_data]  # avoid int64 which can't be written into xls
    gear_data = data_ful[gear_dbc].tolist()
    gear_ratio = main_ratio_dic[vehicle_info]
    gear_data = [number * gear_ratio for number in gear_data]  # avoid int64 which can't be written into xls

    if ((vehicle_info == '途昂') & (temperature_info == '40℃')) | ((vehicle_info == 'AS23') & (temperature_info == '35℃')) \
            | ((vehicle_info == '汉兰达') & (temperature_info == '40℃')):  # for files whose signals from speed box not right
        velocity_dbc = dbc_dic[vehicle_info]['velocity_backup']
        velocity_data = data_ful[velocity_dbc].tolist()
        velocity_data = [velocity/3.6 for velocity in velocity_data]

    # LSS speed correction(optional)
    pedal_zero_index = [1 if pedal < 1 else 0 for pedal in pedal_data]
    break_one_index = [1 if brake > 0 else 0 for brake in brake_data]
    velocity_zero_index = [1 if velocity < 0.1 else 0 for velocity in velocity_data]
    vehicle_still_flag = [pedal_zero_index[i] * break_one_index[i] * velocity_zero_index[i] for i in range(0, len(pedal_zero_index))]
    velocity_zero_offset = np.sum([velocity_data[i] * vehicle_still_flag[i] for i in range(0, len(velocity_data))])/sum(vehicle_still_flag)
    velocity_data_correct = [velocity_data[i] - velocity_zero_offset for i in range(0, len(velocity_data))]
    velocity_data = velocity_data_correct
    print('velocity zero offset: ' + str(round(velocity_zero_offset, 4)))

    # recording frequency check and determine(optional)
    if '时间' in data_ful.columns:
        time_data = data_ful['时间'].tolist()
        start_record_hour = int(time_data[0]/10000)
        start_record_min = int(time_data[0]/100) - start_record_hour * 100
        start_record_second = int(time_data[0]/1) - start_record_hour * 10000 - start_record_min * 100
        start_time = start_record_hour * 3600 + start_record_min * 60 + start_record_second

        end_record_hour = int(time_data[-1]/10000)
        end_record_min = int(time_data[-1]/100) - end_record_hour * 100
        end_record_second = int(time_data[-1]/1) - end_record_hour * 10000 - end_record_min * 100
        end_time = end_record_hour * 3600 + end_record_min * 60 + end_record_second

        fre_cal = len(time_data)/(end_time - start_time)
        fre_int = int(round(fre_cal/10)*10)
        fre = fre_int
    if 'Time (abs)' in data_ful.columns:
        time_data = data_ful['Time (abs)'].tolist()
        start_time = time_data[0]
        end_time = time_data[-1]

        fre_cal = len(time_data)/(end_time - start_time)
        fre_int = int(round(fre_cal/10)*10)
        fre = fre_int
    # recording frequency check and determine ends

    # pedal edges finding：finding segements with pedal > 97
    pedal_filter1 = [1 if pedal_data[i] > 97 else 0 for i in range(0, len(pedal_data))]
    pedal_filter2 = pedal_filter1[1:]
    pedal_filter2.append(pedal_filter1[-1])
    pedal_flag = [pedal_filter2[i] - pedal_filter1[i] for i in range(0, len(pedal_data))]
    pedal_rising_index = [i for i in range(0, len(pedal_data)) if pedal_flag[i] == 1]
    pedal_trailing_index = [i for i in range(0, len(pedal_data)) if pedal_flag[i] == -1]

    # pedal edges filtering: filter segments with pedal hold time < hold_time_min
    pedal_hold_time_filter = [1 if (pedal_trailing_index[i] - pedal_rising_index[i]) > hold_time_min * fre else 0
                              for i in range(0, len(pedal_rising_index))]

    # pedal edges filtering: filter segments with abnormal velocity
    velocity_rising_edge = [velocity_data[i] for i in pedal_rising_index]
    velocity_trailing_edge = [velocity_data[i] for i in pedal_trailing_index]
    velocity_filter = [1 if (velocity_rising_edge[i] < velocity_start_max) & (velocity_trailing_edge[i] > 95/3.6) else 0
                       for i in range(0, len(pedal_rising_index))]

    # filters integration
    final_filter = [pedal_hold_time_filter[i] * velocity_filter[i] for i in range(0, len(pedal_rising_index))]

    # pedal edges filtering
    pedal_rising_index = [pedal_rising_index[i] for i in range(0, len(pedal_rising_index)) if final_filter[i] == 1]
    pedal_trailing_index = [pedal_trailing_index[i] for i in range(0, len(pedal_trailing_index)) if final_filter[i] == 1]

    # brake edges: moments when brake release
    brake_release_index = []
    for i in pedal_rising_index:
        if brake_data[i] > 0:  # with pre speed, trace forward to find the beginning
            for j in range(0, fre*3):
                if brake_data[i+j] == 0:
                    brake_release_index.append(i+j)
                    break
        else:  # without pre speed, same as pedal_rising_index
            brake_release_index.append(i)

    # 0-100 starts: moments where 0.2kph reaches according to speed box
    start_index = []
    for i in pedal_rising_index:
        for j in range(0, fre*20):
            # search from 0.2s before pedal raise to consider: 1.creep due to driver control 2. signal phase inconsistance
            if velocity_data[i+j-int(0.2*fre)] > velocity_start_min:
                start_index.append(i+j-int(0.2*fre))
                break

    # 0-100 ends: moments where 100kph reaches according to speed box
    end_index = []
    for i in brake_release_index:
        if max(velocity_data[i:i+fre*20]) > velocity_end_min:
            for j in range(0, fre*20):
                if velocity_data[i+j] > velocity_end_min:
                    end_index.append(i+j)
                    break
        else:
            for j in range(0, fre*20):
                if velocity_data[i+j] == max(velocity_data[i:i+fre*20-1]):
                    end_index.append(i+j)
                    break

    # temperature measurement module determination
    tem_dic = {'module1': {'pipe_inlet': 'MSChannel01', 'turbine_outlet': 'MSChannel02', 'cooling_outlet': 'MSChannel03',
                           'cooling_to_valve': 'MSChannel04', 'valve_inlet': 'MSChannel05', 'outside': 'MSChannel10'},
               'module2': {'pipe_inlet': 'MSChannel11', 'turbine_outlet': 'MSChannel12', 'cooling_outlet': 'MSChannel13',
                           'cooling_to_valve': 'MSChannel14', 'valve_inlet': 'MSChannel15', 'outside': 'MSChannel20'},
               'module3': {'pipe_inlet': 'MSChannel_1.1', 'turbine_outlet': 'MSChannel_1.2', 'cooling_outlet': 'MSChannel_1.3',
                           'cooling_to_valve': 'MSChannel_1.4', 'valve_inlet': 'MSChannel_1.5', 'outside': 'MSChannel_2.8'}
               }

    if vehicle_info == '途昂':
        tem_dic = {'module1': {'pipe_inlet': 'Channel01', 'turbine_outlet': 'Channel02', 'cooling_outlet': 'Channel03',
                               'cooling_to_valve': 'Channel04', 'valve_inlet': 'Channel05', 'outside': 'Channel10'},
                   'module2': {'pipe_inlet': 'Channel11', 'turbine_outlet': 'Channel12', 'cooling_outlet': 'Channel13',
                               'cooling_to_valve': 'Channel14', 'valve_inlet': 'Channel15', 'outside': 'Channel20'},
                   'module3': {'pipe_inlet': 'Channel_1.1', 'turbine_outlet': 'Channel_1.2', 'cooling_outlet': 'Channel_1.3',
                               'cooling_to_valve': 'Channel_1.4', 'valve_inlet': 'Channel_1.5', 'outside': 'Channel_2.8'}
                   }

    tem_valid_1 = np.average(data_ful[tem_dic['module1']['outside']].tolist())
    tem_valid_2 = np.average(data_ful[tem_dic['module2']['outside']].tolist())
    tem_valid_3 = np.average(data_ful[tem_dic['module3']['outside']].tolist())

    if (tem_valid_1 > 10) | (tem_valid_1 < -10):  # some data has value of  -270C at the beginning due to connecting process
        module = 'module1'
    if (tem_valid_2 > 10) | (tem_valid_2 < -10):
        module = 'module2'
    if (tem_valid_3 > 10) | (tem_valid_3 < -10):
        module = 'module3'

    for keys in tem_dic[module]:
        channel_name = tem_dic[module][keys]
        data = data_ful[channel_name].tolist()
        data = [number * 1.0 for number in data]  # data type transfer: avoid int64 which can't be written to xls
        if np.average(data) < -150:  # some data has value of  -270C at the beginning due to connecting process
            data = [0 for i in data]
        exec(keys + '=data')

    # values extraction
    speed_at_start += [int(speed_data[i]/10)*10 for i in brake_release_index]  # speed at brake release
    velocity_at_start += [round(velocity_data[i]*3.6, 1) for i in start_index]  # velocity at brake release
    velocity_at_end += [round(velocity_data[i]*3.6, 1) for i in end_index]  # velocity at pedal release
    time_for_acc += [(end_index[i] - start_index[i])/fre for i in range(0, len(start_index))]

    pipe_inlet_at_start += [round(pipe_inlet[i], 0) for i in start_index]
    turbine_outlet_at_start += [round(turbine_outlet[i], 0) for i in start_index]
    cooling_outlet_at_start += [round(cooling_outlet[i], 0) for i in start_index]
    cooling_to_valve_at_start += [round(cooling_to_valve[i], 0) for i in start_index]
    valve_inlet_at_start += [round(valve_inlet[i], 0) for i in start_index]
    outside_at_start += [round(outside[i], 0) for i in start_index]

    pipe_inlet_at_end += [round(pipe_inlet[i], 0) for i in end_index]
    turbine_outlet_at_end += [round(turbine_outlet[i], 0) for i in end_index]
    cooling_outlet_at_end += [round(cooling_outlet[i], 0) for i in end_index]
    cooling_to_valve_at_end += [round(cooling_to_valve[i], 0) for i in end_index]
    valve_inlet_at_end += [round(valve_inlet[i], 0) for i in end_index]
    outside_at_end += [round(outside[i], 0) for i in end_index]

    vehicle_number += [vehicle_info for i in start_index]
    temperature_target += [temperature_info for i in start_index]
    frequency_cal += [fre_cal for i in start_index]
    frequency_int += [fre_int for i in start_index]

    start_index_offset = [index for index in start_index]  # instead of using start_index_offset = start_index to decouple two variable, otherwise modify will be taken in both variables
    end_index_offset = [index for index in end_index]
    for order_id in range(0, len(start_index)):
        if (vehicle_info == 'AS23') & (temperature_info == '40℃'):
            offset_time = 1
            start_index_offset[order_id] = start_index[order_id] + int(frequency_int[order_id]*offset_time)
            end_index_offset[order_id] = end_index[order_id] + int(frequency_int[order_id]*offset_time)
            print('enter')
        else:
            start_index_offset[order_id] = start_index[order_id]
            end_index_offset[order_id] = end_index[order_id]

    # process data extraction:
    for i in range(0, len(start_index)):
        torque_data_section.append(torque_data[start_index_offset[i]-lead_time*fre_int: end_index_offset[i]+1])
        speed_data_section.append(speed_data[start_index_offset[i]-lead_time*fre_int: end_index_offset[i]+1])
        velocity_data_section.append([velocity * 3.6 for velocity in velocity_data[start_index[i]-lead_time*fre_int: end_index[i]+1]])
        gear_data_section.append(gear_data[start_index_offset[i]-lead_time*fre_int: end_index_offset[i]+1])
        time_data_section.append([time / fre_int-lead_time for time in range(0, end_index[i]+1-start_index[i]+lead_time*fre_int)])
        order += [count]
        count += 1

# process data packing
process_data_title = ['time', 'torque', 'speed', 'velocity', 'gear']
process_data_assembly = [time_data_section, torque_data_section, speed_data_section, velocity_data_section, gear_data_section]

# selected process data output using old method
# wishing_list = [[i for i in range(0, len(order))]]
# wishing_list = [[4, 12], [8, 16], [22,33, 46], [28, 38, 49]]
# process_data_title = ['time', 'torque', 'speed', 'velocity']
# process_data_column_num = len(process_data_title)
#
# book = xlwt.Workbook(encoding='GB18030', style_compression=0)
# for sheet_id in range(0, len(wishing_list)):
#     sheet = book.add_sheet('procedure data of list' + str(sheet_id))
#
#     for order_id in range(0, len(wishing_list[sheet_id])):
#         torque_data_one_run = torque_data_section[wishing_list[sheet_id][order_id]]
#         speed_data_one_run = speed_data_section[wishing_list[sheet_id][order_id]]
#         velocity_data_one_run = velocity_data_section[wishing_list[sheet_id][order_id]]
#         time_data_one_run = time_data_section[wishing_list[sheet_id][order_id]]
#
#         for column_id in range(0, process_data_column_num):
#             sheet.write(0, order_id * process_data_column_num + column_id, process_data_title[column_id])
#
#         for data_id in range(0, len(torque_data_one_run)):
#             sheet.write(data_id + 1, order_id*process_data_column_num, time_data_one_run[data_id])
#             sheet.write(data_id + 1, order_id*process_data_column_num+1, torque_data_one_run[data_id])
#             sheet.write(data_id + 1, order_id*process_data_column_num+2, speed_data_one_run[data_id])
#             sheet.write(data_id + 1, order_id*process_data_column_num+3, velocity_data_one_run[data_id])
# book.save(r'C:\Users\吕惠加\Desktop\动力衰减\procedure data.xls')


# summary report generating:
# summary data packing
summary_title = ['序号', '车辆', '目标温度', '初始转速', '初始车速', '末车速', '0-100时间',
                 '初始进气口温度', '初始涡轮出口温度', '初始中冷出口温度', '初始中冷到节气门间温度', '初始节气门温度', '初始环境温度',
                 '结束进气口温度', '结束涡轮出口温度', '结束中冷出口温度', '结束中冷到节气门间温度', '结束节气门温度', '结束环境温度', '频率计算值', '频率取整值']
summary_assembly = [order, vehicle_number, temperature_target, speed_at_start, velocity_at_start, velocity_at_end, time_for_acc,
                    pipe_inlet_at_start, turbine_outlet_at_start, cooling_outlet_at_start, cooling_to_valve_at_start, valve_inlet_at_start, outside_at_start,
                    pipe_inlet_at_end, turbine_outlet_at_end, cooling_outlet_at_end, cooling_to_valve_at_end, valve_inlet_at_end, outside_at_end, frequency_cal, frequency_int]
book = xlwt.Workbook(encoding='GB18030', style_compression=0)
sheet = book.add_sheet('summary', cell_overwrite_ok=True)

# title line generation
for i in range(0, len(summary_title)):
    sheet.write(0, i, summary_title[i])

# row adding
row_number = 1

for i in range(0, len(order)):
        for j in range(0, len(summary_assembly)):
            sheet.write(row_number, j, summary_assembly[j][i])
        row_number += 1
book.save(r'C:\Users\吕惠加\Desktop\动力衰减\summary\summary.xls')
#  summary report output finished

# full process data output to xlsx
os.chdir(r'C:\Users\吕惠加\Desktop\动力衰减\summary')
order_num = len(summary_assembly[0])  # total number of test record
expinfo_num = len(summary_assembly)  # total number of expinfo included in summary
signal_num = len(process_data_assembly)  # total number of signal included in process data
column_num = signal_num + 2 + 1  # including 2 expinfo columns and 1 blank column

workbook = xlsxwriter.Workbook("process data_full.xlsx")
worksheet = workbook.add_worksheet()
for order_id in range(0, order_num):
    worksheet.write_column(0, column_num*order_id, summary_title)
    worksheet.write_row(0, column_num*order_id + 2, process_data_title)
    row_num = 0
    for expinfo_id in range(0, expinfo_num):
        worksheet.write_column(row_num, column_num*order_id + 1, [summary_assembly[expinfo_id][order_id]])
        row_num += 1
    for signal_id in range(0, signal_num):
        worksheet.write_column(1, column_num*order_id + 1 + 1 + signal_id, process_data_assembly[signal_id][order_id])
    print(str(order_id) + ' process data output finished')
workbook.close()
# full process data output to xlsx ends

# pkl data dump: for further analysis of data
output_package = {'summary_title': summary_title, 'summary_assembly': summary_assembly,
                  'process_data_title': process_data_title, 'process_data_assembly': process_data_assembly}
os.chdir(r'C:\Users\吕惠加\Desktop\动力衰减\data and figure')
outputfile = open('result_package.pkl', 'wb')
pickle.dump(output_package, outputfile)
outputfile.close()
# pkl data dump ends
