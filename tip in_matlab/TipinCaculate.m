%9.28�Ż���tip in out���߼��㣺
%%NN��tipin data������3offset�����CAN���ٱ�DBOXС�����������Ӧ����
%%eepedalѡ2����1��һЩ��ֹ���ʶ����Ӱ��
%%ʶ�𹤿���֤���ɵ���result��������Ƿ����������˳�򣬲���˳�������ȣ�����������һ���š������Ĺ����϶�����
%%result�����ظ��������ڳ���ʱ�����������������Σ�һ�����Ŀ��ף����ǳ�������ǰ��ģ���ָ������Ƿ�����
%1.���ٶȳ���ɸѡ������holdtime����һ��ʽ�޸ģ���ǿ����Թ���ʶ�������
%2.������legend��pedal��ʾ�������ˣ��ҿɿ�����Щ����δʶ��
%3.�ظ��Ĺ���ʶ���޳�
%4.��ֵ���ٶ�ɸѡ������acc_respond����һ��ʽ�޸ģ����ʹ�ֵ����ʶ������Ÿ߳��ٹ���
%5.ͳһ�˲�ͬ����������ͬһ̤���������ɫ
%6.���������귶Χͳһ��������acc_rangeһ������
%10.1���£�
%1.ɾ������pedalֵ<1���޳����ж���������Ϊ���һЩ�������Ųȹ��ˣ������ŵ��ٶ�������tip in�Ĺ���
%2.��ͼʱ��ǿ���ж�������ӳ�䵽colorindex֮�󣬱����ǣ�1��2��3��4�����ܽ��뻭ͼ���ڣ��޳����󹤿�ʶ���³��򱨴���ֹ
%3.��ͼ�У����������������������60kph�ĳ����б�����ͬһ����̤��Ķ��ʵ�����ߣ������ڵ��Գ���ͱ���ʶ�𵽵ĵ�һ�����󹤿�Ӱ�������ȷ����ʶ�𣬶���������߿����ֶ�ɾ��
%4.����remove_flag��������ֵΪ0�ǲ��Ƴ��ظ������ĳ��������ֵΪ1���Ƴ��ظ������ı�Ҫ����
%10.12����
%���Ӹ�ѡ�����ھ����Ƿ�һ�������λ�仯
%20180209����-cm
%���¿�ͼ��С�仯
%����pedal�źŻ�ͼ
%���¼���ָ�꣺response time & ���ٶȷ�ֵ(0.95)
function varargout = TipinCaculate(varargin)
% TIPINCAC1ULATE MATLAB code for TipinCaculate.fig
%      TipinCaculate, by itself, creates a new TipinCaculate or raises the existing
%      singleton*.
%
%      H = TipinCaculate returns the handle to a new TipinCaculate or the handle to
%      the existing singleton*.
%
%      TipinCaculate('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TIPINCACULATE.M with the given input arguments.
%
%      TIPINCACULATE('Property','Value',...) creates a new TIPINCACULATE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before TipinCaculate_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to TipinCaculate_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help TipinCaculate

% Last Modified by GUIDE v2.5 23-Feb-2018 09:22:59

% Begin initialization code - DO NOT EDIT8
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @TipinCaculate_OpeningFcn, ...
                   'gui_OutputFcn',  @TipinCaculate_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before TipinCaculate is made visible.
function TipinCaculate_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to TipinCaculate (see VARARGIN)

% Choose default command line output for TipinCaculate
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes TipinCaculate wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = TipinCaculate_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in LoadinData.
function LoadinData_Callback(hObject, eventdata, handles)
% hObject    handle to LoadinData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 %%����������ݣ�time��pedal��enginespeed��gear��acc��speed��  
    clc;

    
    global testData FileName 
    [FileName,PathName]=uigetfile({'*.xlsx';'*.xls';'*.csv'},'Select Data');%��ȡԭʼ���ݵ����ֺ�λ��
    strData=strcat(PathName,FileName);%�ļ�λ�ú����ƺϲ�
    [~,~,allData] = xlsread(strData,1);%��ȡstrData���ȫ�����ݣ�strDataֻ���ļ�λ�ú����֣�û�����ݣ�
    [m,~]=size(allData);    %��alldata����
    strTestCell=allData(1,:);%����
    testData=allData(2:m,:); %����������Ҫ��ȫ�����ݣ�ֻ�����֣����ַ� ԭʼΪ18��m-19
    iter=1;
    for i=1:length(strTestCell)
        if ~isnan(strTestCell{i})
            columnCell(iter)=strTestCell(i);%��ѡ����
            iter=iter+1;
        end
    end
%     set(handles.menuTimeData,'string',columnCell);
    set(handles.menuPedalData,'string',columnCell);
    set(handles.menuEngineSpeedData,'string',columnCell);
    set(handles.menuGearData,'string',columnCell);
    set(handles.menuAccData,'string',columnCell);
    set(handles.menuSpeedData,'string',columnCell);
    set(handles.kickdown,'string',columnCell);
    msgbox('ʵ�����ݵ������')
    
%�������ʱ��
function SampleTime_Callback(hObject, eventdata, handles)
% hObject    handle to SampleTime (see GCBO)
global testData splt spt
spt=eval(get(handles.SplTime,'string'));
splt=0:spt:(length(testData)-1)*spt;
msgbox('����ʱ���趨�ɹ�')

%% --- Executes on selection change in menuTimeData.


% --- Executes during object creation, after setting all properties.
function menuTimeData_CreateFcn(hObject, eventdata, handles)
% hObject    handle to menuTimeData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in menuPedalData.
function menuPedalData_Callback(hObject, eventdata, handles)
% hObject    handle to menuPedalData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns menuPedalData contents as cell array
%        contents{get(hObject,'Value')} returns selected item from menuPedalData


% --- Executes during object creation, after setting all properties.
function menuPedalData_CreateFcn(hObject, eventdata, handles)
% hObject    handle to menuPedalData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in menuEngineSpeedData.
function menuEngineSpeedData_Callback(hObject, eventdata, handles)
% hObject    handle to menuEngineSpeedData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns menuEngineSpeedData contents as cell array
%        contents{get(hObject,'Value')} returns selected item from menuEngineSpeedData


% --- Executes during object creation, after setting all properties.
function menuEngineSpeedData_CreateFcn(hObject, eventdata, handles)
% hObject    handle to menuEngineSpeedData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in menuGearData.
function menuGearData_Callback(hObject, eventdata, handles)
% hObject    handle to menuGearData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns menuGearData contents as cell array
%        contents{get(hObject,'Value')} returns selected item from menuGearData


% --- Executes during object creation, after setting all properties.
function menuGearData_CreateFcn(hObject, eventdata, handles)
% hObject    handle to menuGearData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in menuAccData.
function menuAccData_Callback(hObject, eventdata, handles)
% hObject    handle to menuAccData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns menuAccData contents as cell array
%        contents{get(hObject,'Value')} returns selected item from menuAccData


% --- Executes during object creation, after setting all properties.
function menuAccData_CreateFcn(hObject, eventdata, handles)
% hObject    handle to menuAccData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in menuSpeedData.
function menuSpeedData_Callback(hObject, eventdata, handles)
% hObject    handle to menuSpeedData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns menuSpeedData contents as cell array
%        contents{get(hObject,'Value')} returns selected item from menuSpeedData


% --- Executes during object creation, after setting all properties.
function menuSpeedData_CreateFcn(hObject, eventdata, handles)
% hObject    handle to menuSpeedData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%% --- Executes on button press in Caculate.
function Caculate_Callback(hObject, eventdata, handles)
% hObject    handle to Caculate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    global testData splt FileName
gear_flag=get(handles.checkbox1,'Value');    

    %% ���acc��С���˲�
%     acc_index = get(handles.menuAccData,'Value'); %%value�洢��ѡ��������˵���Ŀ�����
% %     time_index = get(handles.menuTimeData,'Value');
%     enginespeed_index = get(handles.menuEngineSpeedData,'Value');
%     gear_index = get(handles.menuGearData,'Value');
%     speed_index = get(handles.menuSpeedData,'Value');
%     pedal_index = get(handles.menuPedalData,'Value');
%     kickdown_index=get(handles.kickdown,'Value');
%     save index_list acc_index enginespeed_index gear_index speed_index pedal_index kickdown_index
    
    load index_list
    pedal_index
    
    kickdown_data=cell2mat(testData(:,kickdown_index));
    assignin('base','testData',testData)
    acc = cell2mat(testData(:,acc_index));
    
%     assignin('base','acc',acc)
    pedal = cell2mat(testData(:,pedal_index));
    if isempty(kickdown_data==1)==0
        pedal(kickdown_data==1)=120;
    end
%     time = cell2mat(testData(:,time_index));
    time = splt';
    enginespeed = cell2mat(testData(:,enginespeed_index));
    gear = cell2mat(testData(:,gear_index));
    speed = cell2mat(testData(:,speed_index));
    % �Լ��ٶȽ��д�������������1�㴦����ʼ������ֹ���ٶȳ�ʼ��Ϊ0��
%     exam1=find(speed<0.1 & pedal==0 & abs(acc)<1); % û��0���ٶε����ݲ�����
%     exam1=find(speed<9999);
%     if isempty(exam1)
%         acc_0=0;
%     else
%         acc_0=mean(acc(exam1));
%     end
    acc_0=-((speed(end)-speed(1))/3.6/(splt(end))-mean(acc)*9.8)/9.8;
    
    acc_0
    assignin('base','acc_0',acc_0);
    acc=acc-acc_0;
    exam=find(acc>1);
    acc(exam)=acc(exam-1);
    acc_origan=acc;
    
    [C1,L1] = wavedec(acc,2,'dmey');%���ܣ����߶�һάС���ֽ⺯��
    [thr1,sorh1,keepapp1] = ddencmp('den','wv',acc);%�����Զ�����С�������ѹ������ֵѡȡ������
    clean1 = wdencmp('gbl',C1,L1,'dmey',2,thr1,sorh1,keepapp1);%����һά���ά�źŵ������ѹ����
    acc=clean1;
    %smooth��ͼ
    acc_smooth=smooth(acc,'sgolay',1);
%     acc_smooth=smooth(acc_smooth,5);
%     acc_smooth = acc;
    
    %% Ŀ�ģ�Ѱ��pedal��������ֱ����һ��pedal����,��Ҫ�ҵ�pedalֹͣ������λ��

pedal1=[pedal(1);pedal(1:end-1)];
ee_pedal=pedal-pedal1;
assignin('base','pedal',pedal)
assignin('base','ee_pedal',ee_pedal)
index1=find(ee_pedal>1)-1; %%acc��һʱ����������1��ʱ����ż���
ee_index1=index1-[index1(1);index1(1:end-1)]; %%����ʱ�̼���������Ԫ�ض�Ӧ��ʱ����Ų�
index2=[index1(1);index1(ee_index1>10);index1(end)]; %%acc���ӵ������������ӵ�ʱ��
 
index(:,1)=index2(1:end-1); %%��ɸ�����߽�
index(:,2)=index2(2:end)-1;
index(end,2)=length(pedal); %%�ڶ������һ����ֵΪ�źŵ�����������ʱ�̣�
index(index(:,2)-index(:,1)<60,:)=[];
%tipin֮ǰ�����ȶ���pedal�ȶ�

for ii=1:1:length(index(:,1))
    pedalcal=pedal(index(ii,1):index(ii,2));
    pedalmax=max(pedalcal);
    index(ii,3)=find(abs(pedalcal-pedalmax)<1.5,1)+index(ii,1)-1;   %��3��Ϊ�ﵽ���pedalֵ��ʼλ��
    
    index(ii,4)=max(acc(index(ii,1):index(ii,1)+60));%������Ϊ����3s�������ٶ�ֵ
    index(ii,5)=max(pedalcal)-pedalcal(1);%��5��Ϊ���pedalֵ����ʼpedalֵ�Ĳ�ֵ
    index(ii,8)=max(find(abs(pedalcal-pedalmax)<1.5))+index(ii,1)-1;%��8��Ϊ�ﵽ���pedalֵ����λ��
    pedalminnd=min(pedal(index(ii,8):index(ii,2)));
    index(ii,6)=find(abs(pedal(index(ii,8):index(ii,2))-pedalminnd)<1.5,1)+index(ii,8)-1;%��6��Ϊ�ﵽ���pedal���
    index(ii,7)=max(find(abs(pedal(index(ii,8):index(ii,2))-pedalminnd)<1.5))+index(ii,8)-1;%��7��Ϊ���pedal������
    index(ii,9)=max(pedalcal);
end

assignin('base','index',index)
assignin('base','time',time)
assignin('base','pedal',pedal)

%%�ɵ�����----------
holdtime=3; %pedal����ʱ���޸�
acc_respond=0.05; %���ٶ����ֵ�޸�
time_range=6;
acc_range=0.5; %��ͼ������������
gear_range=7;
colormap=[0.8431 0.6781 0.2941;0.9061 0.5491 0.2241;0.7291 0.4351 0.2351;0.5181 0.2631 0.1531;0.3101 0.3181 0.1651]; %������ɫ��;
% colormap=[0 1 0;0 0 1;1 0 0;0 0 0; 0.8431 0.6781 0.2941]; %������ɫ��;
remove_flag=1; %Ϊ0ǿ�Ʋ��Ƴ��ظ������������˹�ɸѡ��鹤����Ϊ1����������ͼ
legend_flag=1; %1ʱ��ͼ��legend
%%---------------
%% ʱ����̻��߼��ٹ�С�������pedal����ʼpedal��ֵ��С���ǰ������ݺϲ�
%��ʼpedalС��1����ٶȷ�ֵС��0.05g,̤��仯��ֵС��16��̤������ʱ�����0.4s��pedal�ﵽĿ��pedal����ʱ��С��3s������ȥ��
ee=find( pedal(index(:,1))<0 |  index(:,4)<=acc_respond | index(:,5)<1 | time(index(:,3))-time(index(:,1))>0.5 | time(index(:,8))-time(index(:,3))<holdtime);%|  | ( | );%�����߶ȴ���16,pedal�����½�ʱ��С��3s,��߼��ٶ���Ҫ����0.05.pedal�ȶ�ʱ�䳤��С��4s
assignin('base','ee',ee)

clear global dataindex;
global dataindex index_a
index(ee,:)=[];% 
index_a=index;

for ii=1:1:length(index(:,1))
    dataindex{ii}(:,1)=time(index(ii,1):index(ii,7));
    dataindex{ii}(:,2)=speed(index(ii,1):index(ii,7));
    dataindex{ii}(:,3)=acc(index(ii,1):index(ii,7));
    dataindex{ii}(:,4)=pedal(index(ii,1):index(ii,7));
    dataindex{ii}(:,5)=acc_origan(index(ii,1):index(ii,7));
    dataindex{ii}(:,6)=gear(index(ii,1):index(ii,7));
    dataindex{ii}(:,7)=enginespeed(index(ii,1):index(ii,7));
    dataindex{ii}(:,8)=acc_smooth(index(ii,1):index(ii,7));  
end
assignin('base','dataindex',dataindex)
geardown=[];
upnum=1;
downnum=1;
%% acc�˲�����
global spt
for nn=1:1:size(dataindex,2)   
clear time_cal speed_cal acc_cal pedal_cal

        time_cal=dataindex{nn}(:,1);
        speed_cal=dataindex{nn}(:,2);
        acc_cal=dataindex{nn}(:,3);
        pedal_cal=dataindex{nn}(:,4);
        enginespeed_cal=dataindex{nn}(:,7);
        acc_smooth2=dataindex{nn}(:,8);
         
         % ����delay
         pos_delay(nn)=find(acc_cal>=acc_respond ,1);%����Ҫ����ٶȴ��ڵ���0.05��С��0.08
         rdelay(nn)=time_cal(pos_delay(nn))-time_cal(1);

         % ���� Response timeResponse time
          clear a b
%          [a,b]=findpeaks(acc_cal(1:holdtime*1/spt));%ʱ����3s�ڣ���60����֮�ڣ�a��Сbλ��
%          if isempty(a)
            accpeaks(nn)=0.95*max(acc_cal(1:holdtime*1/spt+1));%��ֵ���ٶ�
%          else
%              accpeaks(nn)=0.95*max(acc_cal(1:holdtime*1/spt));             
%          end
         pos_peaks(nn)=find(acc_cal(1:holdtime*1/spt+1)>=accpeaks(nn),1);
         
         accpeaks(nn)=acc_cal(pos_peaks(nn));
         rtime(nn)=time_cal(pos_peaks(nn))-time_cal(1);
         
%    %���ٶ�smooth�µķ�ֵ����ֵ����
%          clear c d
%          [c,d]=findpeaks(acc_smooth2(1:3*1/spt));%ʱ����3s�ڣ���60����֮�ڣ�c��Сdλ��
%          if isempty(c)
%             accsmpeaks(nn)=max(acc_smooth2(1:3*1/spt));%��ֵ���ٶ�
%             posmpeaks(nn)=find(acc_smooth2(1:3*1/spt)==accsmpeaks(nn),1);
%          else
%              accsmpeak1(nn)=max(c);
%              posmpeaks(nn)=d(find(abs(c-accsmpeak1(nn))<0.03,1));
%              accsmpeaks(nn)=c(find(abs(c-accsmpeak1(nn))<0.03,1));
%          if  posmpeaks(nn)>d(find(c==max(c)))
%              posmpeaks(nn)=find(c==max(c));
%              accsmpeaks(nn)=max(c);
%          end
%          end
         %Ŀ��̤�����
         pedal_max(nn)=max(pedal_cal);
         
         % ���� Pedal change rate ̤���ִ������̤�嵽Ŀ�꿪�ȹ��̵Ŀ��ȱ仯��
         eepalcal=pedal_cal(2:end)-pedal_cal(1:end-1);
         time_pedalrate(nn)=time_cal(find(eepalcal<=0,1))-time_cal(1);
         pos_pedalnd(nn)=find(eepalcal<=0,1);
         pedal_pedalrate(nn)=pedal_cal(pos_pedalnd(nn))-pedal_cal(1);
         pedalrate(nn)=(pedal_pedalrate(nn))/time_pedalrate(nn);
           %����tip in ��������tip inʱ�̷�����ת�٣��Լ�tip���̵���߷�����ת��
         enginespeedst(nn)=enginespeed_cal(1);
         enginespeedpeak(nn)=max(enginespeed_cal(1:pos_peaks(nn)));
         enginespeednd(nn)=enginespeed_cal(pos_peaks(nn));

         delta_enginespeed(nn)=enginespeednd(nn)-enginespeedst(nn);
        
          %�����Ӧ��ǰ���ٶȣ����ճ������ȶ������߳���
         speedst(nn)=speed_cal(1);
         speednd(nn)=max(speed_cal);

         %�����Ӧ��ǰ��pedal
         %pos_pedalnd(nn)=find(eepalcal<=0,1);
         pedalst(nn)=pedal_cal(1);
         pedalnd(nn)=pedal_cal(pos_pedalnd(nn));

         % ����Thump rise rate���ٶȴ�0.05gʱ�̵�0.95����ֵ���ٶȱ仯��
       
         %ͨ�����ֵ���ٶȶԱȾ��Բ�ֵС��0.05g�ĵ�һ��ֵ
         time_thumprate(nn)=time_cal(pos_peaks(nn))-time_cal(pos_delay(nn));
         thumpriserate(nn)=(acc_cal(pos_peaks(nn))-acc_cal(pos_delay(nn)))/time_thumprate(nn);
        
         %pos_peaks(nn)=b(find(a==max(a),1));%��ֵ����λ��

         % ����kick
         clear c d
         [c,d]=findpeaks(-acc_cal(pos_peaks(nn)+1:end));
         c=-c;
         d=d+pos_peaks(nn);
         pos_minpeaks(nn)=d(1);
         tempt=accpeaks(nn)-c(1);
         %�趨����ֵ���ٶȵ���ȵĲ�ֵС��0.05g��Ϊû��kick������0.05g��Ϊ������kick
         if  tempt>0.05
             kick(nn)=tempt;
         else
              kick(nn)=0;
         end

         % ���㵵λ��Ϣ
         clear gear_change
         gear_change=gear(index(nn,1):index(nn,8));

         clear a b
         [a,b]=unique(gear_change);%a��ȡ��λ����bλ����Ϣ
         ss=[a,b];
         ss=sortrows(ss,2);%��λ��Ϣ�������¶��嵵λ��Ϣλ��
         a=ss(:,1);
         b=ss(:,2);
         
         gearst(nn)=gear_change(1);
         gearnd(nn)=min(gear_change);
         gearchange=find(gear_change~=gearst(nn),1);
         if isempty(gearchange)==0
             geartime(nn)=time_cal(gearchange)-time_cal(1);
         else
             geartime(nn)=0;
         end
         
             if min(a)-gearst(nn)<0
                geardown(downnum,1)=gear_change(b(find(a==min(a)))-1);
                geartime(nn)=time_cal(b(find(a==min(a))))-time(index(nn,1));
                geardown(downnum,2)=speed_cal(b(find(a==min(a)))-1);
                geardown(downnum,3)=pedal_cal(b(find(a==min(a)))-1);
                geardown(downnum,4)=gear_change(b(find(a==min(a))));
                downnum=downnum+1;
             end

end

%���ٶȴﵽ0.1g��Ӧʱ��
% assignin('base','dataindex',dataindex)
for kk=1:1:length(dataindex)
    na{kk}=find(dataindex{kk}(:,3)>0.1);
    if isempty(na{kk})
        time01{kk}=0;
    else
   time01{kk}=dataindex{kk}(na{kk}(1,1))-dataindex{kk}(1,1);
    end
end

time01g=cell2mat(time01(1:length(time01)));

clear global tipindata
global tipindata
 tipindata(:,1)=round((speedst+3)/10)*10;
% tipindata(:,1)=speedst;
 tipindata(:,2)=speednd;
 tipindata(:,3)=pedalst;
%  tipindata(:,4)=round(pedalnd/10)*10;
 tipindata(:,4)=round(pedal_max/10)*10;
 tipindata(:,5)=thumpriserate;
 tipindata(:,6)=accpeaks;
 tipindata(:,7)=rtime;
 tipindata(:,8)=rdelay;
tipindata(:,9)=delta_enginespeed;
tipindata(:,10)=pedalrate;
tipindata(:,11)=kick;
tipindata(:,12)=gearst;
tipindata(:,13)= gearnd;
tipindata(:,14)=geartime;
tipindata(:,15)= enginespeedst;
tipindata(:,16)= enginespeednd;
tipindata(:,17)= time01g;

%��ͼ
quest=questdlg('�Ƿ�Ҫ���Ƽ��ٶ�ͼ��','�Ƿ��ͼ��','Yes','No','default');
judge=strcmp(quest,'Yes');

if judge==1
%% ��ȡ��� 
        NN =[20,40,60,80,100];
for ii=1:1:length(NN)
      legend_title{NN(ii)}=[];
      legend_result{NN(ii)}=[];
      H{NN(ii)}=[];
      calNN(NN(ii))=1;
      H3{NN(ii)}=[];
      H4{NN(ii)}=[];
end


for ii=1:1:length(index(:,1))
    
    if length(dataindex{ii}(:,3))>=6/spt;
    acctt=0:spt:(length(dataindex{ii}(1:6/spt,3))-1)*spt;
    dataindex{ii}(6/spt+1:end,:)=[];
    else
    acctt=0:spt:(length(dataindex{ii}(:,3))-1)*spt;
    end

    %% ���ͼ
    
        [m,n]=find(abs(NN-speedst(ii))<5);  %��ͼ���ټ��
        if isempty(m)
            continue
        end
%         if NN(n)~=20
%             continue
%         end
        exam=round(index(ii,9)/10)*10;  %��ͼ���ż��
        if exam == 120
            exam=101;
        end
        if isempty(find(legend_title{NN(n)}(1:end)==exam)) || remove_flag==0 %�����޳��ظ����Ź���
            colorindex=round(index(ii,9)/10)*10/20-1; %�ѣ�40��60��80��100, 101��ӳ�䵽��1��2��3��4��5��
            if (colorindex-1)*(colorindex-2)*(colorindex-3)*(colorindex-4)*(colorindex-5)==0
            figure(NN(n))
            title(strcat('����',num2str(NN(n)),'km/h'),'fontsize',14,'fontname','΢���ź�')
            if gear_flag==1
                if calNN(NN(n))==1
                [AX{NN(n)},H1,H2]=plotyy(acctt,dataindex{ii}(:,3),acctt,dataindex{ii}(:,6));
                else
                [AX{NN(n)},H1,H2]=plotyy(AX{NN(n)},acctt,dataindex{ii}(:,3),acctt,dataindex{ii}(:,6));%��Ϊ�м��ٶȣ����ź͵�λ3��y����Ӧ����������ϵ��������ָ��������ử��pedal������
                end

                hold on
                
                H{NN(n)}=[H{NN(n)};H1];
                h3{NN(n)}(calNN(NN(n)))=plot(AX{NN(n)}(1),acctt(pos_delay(ii)),dataindex{ii}(pos_delay(ii),3),'r^','linewidth',4);
                hold on
                h4{NN(n)}(calNN(NN(n)))=plot(AX{NN(n)}(1),acctt(pos_peaks(ii)),dataindex{ii}(pos_peaks(ii),3),'ro','linewidth',4);
                hold on

               
                set(AX{NN(n)}(1),'YLim',[-0.1 (acc_range+0.1)*2-0.1],'XLim',[0 time_range],'Ytick',[-0.1:0.1:acc_range],'fontsize',12,'ycolor','black');
                set(AX{NN(n)}(2),'YLim',[-gear_range gear_range],'XLim',[0 time_range],'Ytick',0:1:gear_range,'fontsize',12,'ycolor','black');
                set(get(AX{NN(n)}(1),'YLabel'),'string','���ٶ�/g','fontsize',14,'color','black','fontname','΢���ź�');
                set(get(AX{NN(n)}(1),'XLabel'),'string','ʱ��/s','fontsize',14,'color','black','fontname','΢���ź�');
                set(get(AX{NN(n)}(2),'YLabel'),'string','��λ','fontsize',14,'color','black','fontname','΢���ź�');
                set(H1,'color',colormap(colorindex,:),'linewidth',4);
                set(H2,'color',colormap(colorindex,:),'linewidth',4,'linestyle','-.');
                
                
                lengendtittle=min(round(index(ii,9)/10)*10,101);
                legend_title{NN(n)}=[legend_title{NN(n)}(1:end);lengendtittle];
                oo=strcat(num2str(lengendtittle),'%');
%                if isempty(legend_result{NN(n)})
%                    legend_result{NN(n)}=oo;
%                    legend([H{NN(n)};h3{NN(n)}(1);h4{NN(n)}(1)],[legend_result(NN(n));'0.02g';'0.95maxacc'] ,'Location','NorthEast','color','white')%
%                else
                   legend_result{NN(n)}=[legend_result{NN(n)};{oo}];
                   if legend_flag==1
                   legend([H{NN(n)};h3{NN(n)}(1);h4{NN(n)}(1)],[legend_result{NN(n)};'0.02g';'0.95maxacc'] ,'Location','NorthEast','color','white')%ǰһ���������ڱ�ʾҪ�ӱ�ǩ���ߣ�����������Ϊ��ǩ����
                   end
%                end
                if calNN(NN(n))==1 %AX3������pedal�����������ޱ�ǩ
                AX3(NN(n))=axes('Tag','pedal','position',get(gca,'position'),'xlim',[0 time_range],'ylim',[0,120],'Color','none','xtick',[],'ytick',[]);
                end
                dataindex{ii}(dataindex{ii}(:,4)>100,4)=101;
                H5=plot(AX3(NN(n)),acctt,dataindex{ii}(:,4),'Color',colormap(colorindex,:),'LineWidth',2,'linestyle','--');  
                hold on
                set(AX3(NN(n)),'YLim',[0 101],'XLim',[0 time_range],'visible','off');
            else
                 dataindex{ii}(dataindex{ii}(:,4)==120,4)=101;
                if calNN(NN(n))==1
                   [AX{NN(n)},H1,H2]=plotyy(acctt,dataindex{ii}(:,3),acctt,dataindex{ii}(:,4));
                else
                   [AX{NN(n)},H1,H2]=plotyy(AX{NN(n)},acctt,dataindex{ii}(:,3),acctt,dataindex{ii}(:,4));            
                end
                hold on
                set(AX{NN(n)}(1),'YLim',[0 acc_range],'XLim',[0 time_range],'Ytick',[0:0.1:acc_range],'fontsize',12,'ycolor','black');
                set(get(AX{NN(n)}(1),'YLabel'),'string','���ٶ�/g','fontsize',14,'color','black','fontname','΢���ź�');
                set(get(AX{NN(n)}(1),'XLabel'),'string','ʱ��/s','fontsize',14,'color','black','fontname','΢���ź�');
                set(AX{NN(n)}(2),'XLim',[0 time_range],'YLim',[0 101],'Ytick',[0 101],'fontsize',12,'ycolor','black');%
                set(get(AX{NN(n)}(2),'YLabel'),'string','����̤��','fontsize',14,'color','black','fontname','΢���ź�');
                set(H1,'color',colormap(colorindex,:),'linewidth',4);
                set(H2,'color',colormap(colorindex,:),'linewidth',2,'linestyle','-.');
                
                h3{NN(n)}(calNN(NN(n)))=plot(AX{NN(n)}(1),acctt(pos_delay(ii)),dataindex{ii}(pos_delay(ii),3),'r^','linewidth',4);
                hold on
                h4{NN(n)}(calNN(NN(n)))=plot(AX{NN(n)}(1),acctt(pos_peaks(ii)),dataindex{ii}(pos_peaks(ii),3),'ro','linewidth',4);
                hold on
                


                H{NN(n)}(calNN(NN(n)))=H1;%ǰ��H����ΪH=[]�������飬���ֱ�Ӵ�H1��ȥ��������double������line����������Ϊline��handle

                lengendtittle=min(round(index(ii,9)/10)*10,101);
                legend_title{NN(n)}=[legend_title{NN(n)}(1:end);lengendtittle];
                legend_result{NN(n)}{calNN(NN(n))}=strcat(num2str(lengendtittle),'%');
                assignin('base','legend_result',legend_result{NN(n)})
                assignin('base','H',H{NN(n)})
                if legend_flag==1
                legend([H{NN(n)},h3{NN(n)}(1),h4{NN(n)}(1)],[legend_result{NN(n)},'0.02g','0.95maxacc'])%,'Location','NorthEast','color','white')
                end
            end

            calNN(NN(n))=calNN(NN(n))+1;
            end
        end
end


    filepath=uigetdir({},'ѡ���ļ���');  
    if isempty(filepath)==0
       assignin('base','filepath',filepath)
    for pp=1:1:length(NN)

         saveas(figure(NN(pp)),strcat(filepath,'\',num2str(NN(pp)),'_',FileName(1:end-4)),'fig')
         saveas(figure(NN(pp)),strcat(filepath,'\',num2str(NN(pp)),'_',FileName(1:end-4)),'bmp')
    end
    end
end
msgbox('�������')




% --- Executes on button press in SaveResults.
function SaveResults_Callback(hObject, eventdata, handles)
% hObject    handle to SaveResults (see GCBO)
    global tipindata

    [filename, pathname] = uiputfile({'*.xls'}, 'Save as');
    str=strcat(pathname,filename);
    if (filename-0)>eps
       
 xlswrite(filename,{'�ȳ���'},'Results','A1');
%  tipindata(:,1)=round(speedst/10)*10;
 xlswrite(filename,{'���ճ���'},'Results','B1');
%  tipindata(:,2)=speednd;
 xlswrite(filename,{'�ȳ���̤�忪��'},'Results','C1')
%  tipindata(:,3)=pedalst;
 xlswrite(filename,{'Ŀ�꿪��'},'Results','D1')
%  tipindata(:,4)=round(pedalnd/10)*10;
 xlswrite(filename,{'���ٶȱ仯��'},'Results','E1');
%  tipindata(:,5)=thumpriserate;
 xlswrite(filename,{'���ٶȷ�ֵ'},'Results','F1')
%  tipindata(:,6)=accpeaks;
 xlswrite(filename,{'��Ӧʱ��'},'Results','G1')
%  tipindata(:,7)=rtime;
 xlswrite(filename,{'delay'},'Results','H1')
%  tipindata(:,8)=rdelay;
xlswrite(filename,{'delta_enginespeed'},'Results','I1')
% tipindata(:,9)=delta_enginespeed;
xlswrite(filename,{'pedal�仯��'},'Results','J1')
% tipindata(:,10)=pedalrate;
xlswrite(filename,{'kick'},'Results','K1')
% tipindata(:,11)=kick;
xlswrite(filename,{'gearst'},'Results','L1')
% tipindata(:,12)=gearst;
xlswrite(filename,{'gearnd'},'Results','M1')
% tipindata(:,13)= gearnd;
xlswrite(filename,{'����ʱ��'},'Results','N1')
% tipindata(:,14)=geartime;
xlswrite(filename,{'enginespeedst'},'Results','O1')
% tipindata(:,15)= enginespeedst;
xlswrite(filename,{'enginespeednd'},'Results','P1')
% tipindata(:,16)= enginespeednd;
xlswrite(filename,{'time0.1g'},'Results','Q1')
% tipindata(:,17)= time01g;
vel_standard=[20,40,60,80,100];
ped_standard=[40,60,80,100,101];
[row,col]=size(tipindata);
i=1;
while i<=row
    if min(abs(vel_standard-tipindata(i,1)))~=0 || min(abs(ped_standard-tipindata(i,4)))~=0
        tipindata(i,:)=[];
        row=row-1;
        i=i-1;
    end
    i=i+1;
end
xlswrite(filename,tipindata,'Results','A2')
     msgbox('���ݵ������')
    end
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in SaveData.
function SaveData_Callback(hObject, eventdata, handles)
% hObject    handle to SaveData (see GCBO)
global dataindex index_a
[filename, pathname] = uiputfile({'*.xls'}, 'Save as');
    str=strcat(pathname,filename);
    if (filename-0)>eps
  for ii=1:1:length(index_a(:,1))
  dataindex{ii}(:,8)=[];     
 xlswrite(filename,{'Time'},strcat('Data',num2str(ii)),'A1');
 xlswrite(filename,{'Speed'},strcat('Data',num2str(ii)),'B1');
 xlswrite(filename,{'Acc'},strcat('Data',num2str(ii)),'C1');
 xlswrite(filename,{'Pedal'},strcat('Data',num2str(ii)),'D1');
 xlswrite(filename,{'Acc_origan'},strcat('Data',num2str(ii)),'E1');
 xlswrite(filename,{'Gear'},strcat('Data',num2str(ii)),'F1');
 xlswrite(filename,{'Enginespeed'},strcat('Data',num2str(ii)),'G1');
 xlswrite(filename,dataindex{ii},strcat('Data',num2str(ii)),'A2');   
  end
  msgbox('���ݵ������')
    end
    
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in PlotResults.
function PlotResults_Callback(hObject, eventdata, handles)
% hObject    handle to PlotResults (see GCBO)
global tipindata
datasort=sortrows(tipindata,1);
xx=unique(datasort(:,1));
[p,q]=hist(datasort(:,1),xx);
p=[0,p];
sump=0;

for k=1:1:length(p)-1
   sump=sump+p(k);
    figure(6)
    plot(datasort((sump+1):(sump+p(k+1)),4),datasort((sump+1):(sump+p(k+1)),6),'Color',[0.01*randi([0,100],1),0.01*randi([0,100],1),0.01*randi([0,100],1)],'LineWidth',2.5)
    hold on
    xlabel('pedal')
    ylabel('��ֵ���ٶ�')
    legend(num2str(q),0)
    figure(7)
    plot(datasort((sump+1):(sump+p(k+1)),4),datasort((sump+1):(sump+p(k+1)),17),'Color',[0.01*randi([0,100],1),0.01*randi([0,100],1),0.01*randi([0,100],1)],'LineWidth',2.5)
    hold on
    xlabel('pedal')
    ylabel('�ﵽ0.1g���ٶ�����ʱ��')
    legend(num2str(q),0)
    figure(8)
    plot(datasort((sump+1):(sump+p(k+1)),4),datasort((sump+1):(sump+p(k+1)),7),'Color',[0.01*randi([0,100],1),0.01*randi([0,100],1),0.01*randi([0,100],1)],'LineWidth',2.5)
    hold on
    xlabel('pedal')
    ylabel('�ﵽ��ֵ��Ӧʱ��')
    legend(num2str(q),0)
    figure(9)
    plot(datasort((sump+1):(sump+p(k+1)),4)-datasort((sump+1):(sump+p(k+1)),3),datasort((sump+1):(sump+p(k+1)),6),'Color',[0.01*randi([0,100],1),0.01*randi([0,100],1),0.01*randi([0,100],1)],'LineWidth',2.5)
    hold on
    ylabel('��ֵ���ٶ�')
    xlabel('delta_pedal')
    legend(num2str(q),0)    
end
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)





% --- Executes during object creation, after setting all properties.
function SplTime_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SplTime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function SplTime_Callback(hObject, eventdata, handles)
% hObject    handle to SplTime (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of SplTime as text
%        str2double(get(hObject,'String')) returns contents of SplTime as a double


% --- Executes on button press in SampleTime.




% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox1


% --- Executes when uipanel1 is resized.
function uipanel1_ResizeFcn(hObject, eventdata, handles)
% hObject    handle to uipanel1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes when uipanel2 is resized.
function uipanel2_ResizeFcn(hObject, eventdata, handles)
% hObject    handle to uipanel2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes when uipanel4 is resized.
function uipanel4_ResizeFcn(hObject, eventdata, handles)
% hObject    handle to uipanel4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in kickdown.
function kickdown_Callback(hObject, eventdata, handles)
% hObject    handle to kickdown (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns kickdown contents as cell array
%        contents{get(hObject,'Value')} returns selected item from kickdown


% --- Executes during object creation, after setting all properties.
function kickdown_CreateFcn(hObject, eventdata, handles)
% hObject    handle to kickdown (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
