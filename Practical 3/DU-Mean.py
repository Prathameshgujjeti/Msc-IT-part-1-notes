################################################################
# -*- coding: utf-8 -*-
################################################################
import sys
import pandas as pd
################################################################
InputFileName='IP_DATA_CORE.csv'
OutputFileName='Retrieve_Router_Location.csv'
################################################################
if sys.platform == 'linux': 
    Base=os.path.expanduser('~') + '/VKHCG'
else:
    Base='C:/Users/prath/OneDrive/Desktop/Msc-IT Practicals/Data Science/Practical 3'
print('################################')
print('Working Base :',Base, ' using ', sys.platform)
print('################################')
################################################################
sFileName=Base + '/' + InputFileName
print('Loading :',sFileName)
IP_DATA_ALL=pd.read_csv(sFileName,header=0,low_memory=False,
  usecols=['Country','Place Name','Latitude','Longitude'], encoding="latin-1")
IP_DATA_ALL.rename(columns={'Place Name': 'Place_Name'}, inplace=True)
AllData=IP_DATA_ALL[['Country', 'Place_Name','Latitude']]
print(AllData)
MeanData=AllData.groupby(['Country', 'Place_Name'])['Latitude'].mean()
print(MeanData)
################################################################