from nptdms import TdmsFile
# from saic_project_django.common import mongo
import os
import mdfreader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse  # pylint: disable=wrong-import-position

tdms_file_path = os.path.join(r'D:\data sample\BMWX3_N20B_AT_LBVTZ6103KSP40554__100%_20190122_by_lst\Data\Comfort\TipInOut\40%_20kph', "data.tdms")
tdms_file = TdmsFile(tdms_file_path)

data_dict = {}
air_flow = []
afr = []
fuel_rate = []
for items in tdms_file.objects:
    if tdms_file.objects[items].has_data:
        group = tdms_file.objects[items].group
        channel = tdms_file.objects[items].channel
        signal = tdms_file.object(group, channel)
        data = signal.data
        if group not in data_dict:
            data_dict[group] = {}
        if isinstance(data, list):
            data_dict[group][channel] = data
        else:
            data_dict[group][channel] = data.tolist()
        if channel == 'Diag':
            quo = [x//65536 for x in data]
            res = [np.mod(x, 65536) for x in data]
            for i in range(0, len(quo)):
                if int(quo[i]) == 16:
                    air_flow.append(0.01*res[i])
                    if i > 0:
                        afr.append(afr[i-1])
                    else:
                        afr.append(2)
                if int(quo[i]) == 68:
                    afr.append(3.0517578e-5*res[i]*14.7)
                    if i > 0:
                        air_flow.append(air_flow[i-1])
                    else:
                        air_flow.append(0)
            fuel_rate = [air_flow[i]/afr[i] if afr[i] <= 1.9*14.7 else 0 for i in range(0, len(air_flow))]
            fuel_rate = [x*3.6/0.75 for x in fuel_rate]
            data_dict[group]['Diag_processed'] = fuel_rate  #L/h
            data_dict[group]['air_flow'] = air_flow
            data_dict[group]['afr'] = afr

writer = pd.ExcelWriter(os.path.join(r'D:\data sample\BMWX3_N20B_AT_LBVTZ6103KSP40554__100%_20190122_by_lst\Data', 'data.xlsx'))
df_list = []
for group in data_dict:
    if group == 'CAN':
        df_list.append(pd.DataFrame(data_dict[group]))
        df_list[-1].to_excel(writer, sheet_name=group)
writer.save()