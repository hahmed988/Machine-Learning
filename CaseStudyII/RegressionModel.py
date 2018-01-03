# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:44:17 2017

@author: hahmed
"""

import pandas as pd
import os
path = 'C:\\MOOC\\PHD_Scripts'
os.getcwd()
os.chdir(path)
import codecs
import seaborn as sns
import matplotlib.pyplot as plt

########################################################################################################
#Weather Data
#Read Weather Data from Separate sheet
xl = pd.read_excel('WeatherData.xlsx', sheetname=None)
xl_dict = {}
sheetname_list = ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']
for sheet in sheetname_list:
    xl_dict[sheet] = pd.read_excel('WeatherData.xlsx', sheetname=sheet)
    
#Create separate dataframes of each for preprocessing  
Weather_2009 = xl_dict['2009']
Weather_2010 = xl_dict['2010']
Weather_2011 = xl_dict['2011']
Weather_2012 = xl_dict['2012']
Weather_2013 = xl_dict['2013']
Weather_2014 = xl_dict['2014']
Weather_2015 = xl_dict['2015']
Weather_2016 = xl_dict['2016']

#Correct the years in each sheet with the correct year
Weather_2009['Year'] = 2009
Weather_2010['Year'] = 2010
Weather_2011['Year'] = 2011
Weather_2012['Year'] = 2012
Weather_2013['Year'] = 2013
Weather_2014['Year'] = 2014
Weather_2015['Year'] = 2015
Weather_2016['Year'] = 2016

#PreProcess All yearly Data from 2009 to 2016 to perform aggregation based on months
#Feature Engineer A new variable to figure out Abnormal weather conditions on a particular
Weather_2016['EventOrNot'] = Weather_2016['WeatherEvent'].apply(lambda x: 0 if pd.isnull(x) else len(x.split(",")))
Weather_2016 = Weather_2016.groupby(['Month']).mean().reset_index()
Weather_2016['Year'] = Weather_2016.Year.astype(int)
Weather_2016['Day'] = Weather_2016.Day.astype(int)
Weather_2016 = Weather_2016.drop('Day', 1)

import numpy as np
Weather_2009['EventOrNot'] = Weather_2009['WeatherEvent'].apply(lambda x: 0 if pd.isnull(x) else len(x.split(",")))
#Weather_2009['EventOrNot'] = Weather_2009['WeatherEvent'].apply(lambda x: len(x.split(","), axis=1)
Weather_2009.head()

Weather_2009 = Weather_2009.groupby(['Month']).mean().reset_index()
Weather_2009['Year'] = Weather_2009.Year.astype(int)
Weather_2009['Day'] = Weather_2009.Day.astype(int)
Weather_2009 = Weather_2009.drop('Day', 1)
Weather_2009.head()

Weather_2010['EventOrNot'] = Weather_2010['WeatherEvent'].apply(lambda x: 0 if pd.isnull(x) else len(x.split(",")))
Weather_2010.head()
Weather_2010 = Weather_2010.groupby(['Month']).mean().reset_index()
Weather_2010.head()

Weather_2010['Year'] = Weather_2010.Year.astype(int)
Weather_2010['Day'] = Weather_2010.Day.astype(int)
Weather_2010 = Weather_2010.drop('Day', 1)
Weather_2010

Weather_2011['EventOrNot'] = Weather_2011['WeatherEvent'].apply(lambda x: 0 if pd.isnull(x) else len(x.split(",")))
Weather_2011 = Weather_2011.groupby(['Month']).mean().reset_index()
Weather_2011['Year'] = Weather_2011.Year.astype(int)
Weather_2011['Day'] = Weather_2011.Day.astype(int)
Weather_2011 = Weather_2011.drop('Day', 1)

print(Weather_2012.isnull().values.any())
Weather_2012 = Weather_2012.replace({'-': 0}, regex=True)

Weather_2012['EventOrNot'] = Weather_2012['WeatherEvent'].apply(lambda x: 0 if pd.isnull(x) else len(x.split(",")))
Weather_2012.head()

Weather_2012 = Weather_2012.drop('WeatherEvent', 1)
Weather_2012.head()

Weather_2012 = Weather_2012.groupby(['Month']).mean().reset_index()

Weather_2012['Year'] = Weather_2012.Year.astype(int)
Weather_2012['Day'] = Weather_2012.Day.astype(int)
Weather_2012 = Weather_2012.drop('Day', 1)

Weather_2013['EventOrNot'] = Weather_2013['WeatherEvent'].apply(lambda x: 0 if pd.isnull(x) else len(x.split(",")))
Weather_2013 = Weather_2013.groupby(['Month']).mean().reset_index()
Weather_2013['Year'] = Weather_2013.Year.astype(int)
Weather_2013['Day'] = Weather_2013.Day.astype(int)
Weather_2013 = Weather_2013.drop('Day', 1)

Weather_2014['EventOrNot'] = Weather_2014['WeatherEvent'].apply(lambda x: 0 if pd.isnull(x) else len(x.split(",")))
print(Weather_2014.isnull().values.any())
#Replace - and T values with 0.
Weather_2014 = Weather_2014.replace({'-': 0}, regex=True)
Weather_2014 = Weather_2014.replace({'T': 0}, regex=True)
print(Weather_2014.isnull().sum().sum())
Weather_2014 = Weather_2014.drop('WeatherEvent', 1)

#Remove rows with value as high from the 2014 data
Weather_2014 = Weather_2014[Weather_2014[u'Temp high (°C)'] != 'high']
Weather_2014.head()

Weather_2014[u'Temp avg (°C)'] = Weather_2014[u'Temp avg (°C)'].astype(str).astype(float)
Weather_2014 = Weather_2014.groupby(['Month']).mean().reset_index()
Weather_2014['Year'] = Weather_2014.Year.astype(int)
Weather_2014['Day'] = Weather_2014.Day.astype(int)
Weather_2014 = Weather_2014.drop('Day', 1)

Weather_2015['EventOrNot'] = Weather_2015['WeatherEvent'].apply(lambda x: 0 if pd.isnull(x) else len(x.split(",")))
Weather_2015 = Weather_2015.groupby(['Month']).mean().reset_index()
Weather_2015['Year'] = Weather_2015.Year.astype(int)
Weather_2015['Day'] = Weather_2015.Day.astype(int)
Weather_2015 = Weather_2015.drop('Day', 1)

#Selecting Subset of Columns from Weather Table 
Weather_2009_Sub  = Weather_2009[['Month','Year', 'EventOrNot',   u'Temp avg (°C)']]
Weather_2010_Sub  = Weather_2010[['Month','Year', 'EventOrNot',   u'Temp avg (°C)']]
Weather_2011_Sub  = Weather_2011[['Month','Year', 'EventOrNot',   u'Temp avg (°C)']]
Weather_2012_Sub  = Weather_2012[['Month','Year', 'EventOrNot',   u'Temp avg (°C)']]
Weather_2013_Sub  = Weather_2013[['Month','Year', 'EventOrNot',   u'Temp avg (°C)']]
Weather_2014_Sub  = Weather_2014[['Month','Year', 'EventOrNot',   u'Temp avg (°C)']]
Weather_2015_Sub  = Weather_2015[['Month','Year', 'EventOrNot',   u'Temp avg (°C)']]
Weather_2016_Sub  = Weather_2016[['Month','Year', 'EventOrNot',   u'Temp avg (°C)']]


#Convert Month to Integer values for easier merge with Train and Macroeconomic Data
Weather_2016_Sub['Month'] = Weather_2016_Sub['Month'].apply(lambda x: 1 if x == 'Jan' else x)
Weather_2016_Sub['Month'] = Weather_2016_Sub['Month'].apply(lambda x: 2 if x == 'Feb' else x)
Weather_2016_Sub['Month'] = Weather_2016_Sub['Month'].apply(lambda x: 3 if x == 'Mar' else x)
Weather_2016_Sub['Month'] = Weather_2016_Sub['Month'].apply(lambda x: 4 if x == 'Apr' else x)
Weather_2016_Sub['Month'] = Weather_2016_Sub['Month'].apply(lambda x: 5 if x == 'May' else x)
Weather_2016_Sub['Month'] = Weather_2016_Sub['Month'].apply(lambda x: 6 if x == 'Jun' else x)
Weather_2016_Sub['Month'] = Weather_2016_Sub['Month'].apply(lambda x: 7 if x == 'Jul' else x)
Weather_2016_Sub['Month'] = Weather_2016_Sub['Month'].apply(lambda x: 8 if x == 'Aug' else x)
Weather_2016_Sub['Month'] = Weather_2016_Sub['Month'].apply(lambda x: 9 if x == 'Sep' else x)
Weather_2016_Sub['Month'] = Weather_2016_Sub['Month'].apply(lambda x: 10 if x == 'Oct' else x)
Weather_2016_Sub['Month'] = Weather_2016_Sub['Month'].apply(lambda x: 11 if x == 'Nov' else x)
Weather_2016_Sub['Month'] = Weather_2016_Sub['Month'].apply(lambda x: 12 if x == 'Dec' else x)


Weather_2009_Sub['Month'] = Weather_2009_Sub['Month'].apply(lambda x: 1 if x == 'Jan' else x)
Weather_2009_Sub['Month'] = Weather_2009_Sub['Month'].apply(lambda x: 2 if x == 'Feb' else x)
Weather_2009_Sub['Month'] = Weather_2009_Sub['Month'].apply(lambda x: 3 if x == 'Mar' else x)
Weather_2009_Sub['Month'] = Weather_2009_Sub['Month'].apply(lambda x: 4 if x == 'Apr' else x)
Weather_2009_Sub['Month'] = Weather_2009_Sub['Month'].apply(lambda x: 5 if x == 'May' else x)
Weather_2009_Sub['Month'] = Weather_2009_Sub['Month'].apply(lambda x: 6 if x == 'Jun' else x)
Weather_2009_Sub['Month'] = Weather_2009_Sub['Month'].apply(lambda x: 7 if x == 'Jul' else x)
Weather_2009_Sub['Month'] = Weather_2009_Sub['Month'].apply(lambda x: 8 if x == 'Aug' else x)
Weather_2009_Sub['Month'] = Weather_2009_Sub['Month'].apply(lambda x: 9 if x == 'Sep' else x)
Weather_2009_Sub['Month'] = Weather_2009_Sub['Month'].apply(lambda x: 10 if x == 'Oct' else x)
Weather_2009_Sub['Month'] = Weather_2009_Sub['Month'].apply(lambda x: 11 if x == 'Nov' else x)
Weather_2009_Sub['Month'] = Weather_2009_Sub['Month'].apply(lambda x: 12 if x == 'Dec' else x)


Weather_2010_Sub['Month'] = Weather_2010_Sub['Month'].apply(lambda x: 1 if x == 'Jan' else x)
Weather_2010_Sub['Month'] = Weather_2010_Sub['Month'].apply(lambda x: 2 if x == 'Feb' else x)
Weather_2010_Sub['Month'] = Weather_2010_Sub['Month'].apply(lambda x: 3 if x == 'Mar' else x)
Weather_2010_Sub['Month'] = Weather_2010_Sub['Month'].apply(lambda x: 4 if x == 'Apr' else x)
Weather_2010_Sub['Month'] = Weather_2010_Sub['Month'].apply(lambda x: 5 if x == 'May' else x)
Weather_2010_Sub['Month'] = Weather_2010_Sub['Month'].apply(lambda x: 6 if x == 'Jun' else x)
Weather_2010_Sub['Month'] = Weather_2010_Sub['Month'].apply(lambda x: 7 if x == 'Jul' else x)
Weather_2010_Sub['Month'] = Weather_2010_Sub['Month'].apply(lambda x: 8 if x == 'Aug' else x)
Weather_2010_Sub['Month'] = Weather_2010_Sub['Month'].apply(lambda x: 9 if x == 'Sep' else x)
Weather_2010_Sub['Month'] = Weather_2010_Sub['Month'].apply(lambda x: 10 if x == 'Oct' else x)
Weather_2010_Sub['Month'] = Weather_2010_Sub['Month'].apply(lambda x: 11 if x == 'Nov' else x)
Weather_2010_Sub['Month'] = Weather_2010_Sub['Month'].apply(lambda x: 12 if x == 'Dec' else x)

Weather_2011_Sub['Month'] = Weather_2011_Sub['Month'].apply(lambda x: 1 if x == 'Jan' else x)
Weather_2011_Sub['Month'] = Weather_2011_Sub['Month'].apply(lambda x: 2 if x == 'Feb' else x)
Weather_2011_Sub['Month'] = Weather_2011_Sub['Month'].apply(lambda x: 3 if x == 'Mar' else x)
Weather_2011_Sub['Month'] = Weather_2011_Sub['Month'].apply(lambda x: 4 if x == 'Apr' else x)
Weather_2011_Sub['Month'] = Weather_2011_Sub['Month'].apply(lambda x: 5 if x == 'May' else x)
Weather_2011_Sub['Month'] = Weather_2011_Sub['Month'].apply(lambda x: 6 if x == 'Jun' else x)
Weather_2011_Sub['Month'] = Weather_2011_Sub['Month'].apply(lambda x: 7 if x == 'Jul' else x)
Weather_2011_Sub['Month'] = Weather_2011_Sub['Month'].apply(lambda x: 8 if x == 'Aug' else x)
Weather_2011_Sub['Month'] = Weather_2011_Sub['Month'].apply(lambda x: 9 if x == 'Sep' else x)
Weather_2011_Sub['Month'] = Weather_2011_Sub['Month'].apply(lambda x: 10 if x == 'Oct' else x)
Weather_2011_Sub['Month'] = Weather_2011_Sub['Month'].apply(lambda x: 11 if x == 'Nov' else x)
Weather_2011_Sub['Month'] = Weather_2011_Sub['Month'].apply(lambda x: 12 if x == 'Dec' else x)

Weather_2012_Sub['Month'] = Weather_2012_Sub['Month'].apply(lambda x: 1 if x == 'Jan' else x)
Weather_2012_Sub['Month'] = Weather_2012_Sub['Month'].apply(lambda x: 2 if x == 'Feb' else x)
Weather_2012_Sub['Month'] = Weather_2012_Sub['Month'].apply(lambda x: 3 if x == 'Mar' else x)
Weather_2012_Sub['Month'] = Weather_2012_Sub['Month'].apply(lambda x: 4 if x == 'Apr' else x)
Weather_2012_Sub['Month'] = Weather_2012_Sub['Month'].apply(lambda x: 5 if x == 'May' else x)
Weather_2012_Sub['Month'] = Weather_2012_Sub['Month'].apply(lambda x: 6 if x == 'Jun' else x)
Weather_2012_Sub['Month'] = Weather_2012_Sub['Month'].apply(lambda x: 7 if x == 'Jul' else x)
Weather_2012_Sub['Month'] = Weather_2012_Sub['Month'].apply(lambda x: 8 if x == 'Aug' else x)
Weather_2012_Sub['Month'] = Weather_2012_Sub['Month'].apply(lambda x: 9 if x == 'Sep' else x)
Weather_2012_Sub['Month'] = Weather_2012_Sub['Month'].apply(lambda x: 10 if x == 'Oct' else x)
Weather_2012_Sub['Month'] = Weather_2012_Sub['Month'].apply(lambda x: 11 if x == 'Nov' else x)
Weather_2012_Sub['Month'] = Weather_2012_Sub['Month'].apply(lambda x: 12 if x == 'Dec' else x)

Weather_2013_Sub['Month'] = Weather_2013_Sub['Month'].apply(lambda x: 1 if x == 'Jan' else x)
Weather_2013_Sub['Month'] = Weather_2013_Sub['Month'].apply(lambda x: 2 if x == 'Feb' else x)
Weather_2013_Sub['Month'] = Weather_2013_Sub['Month'].apply(lambda x: 3 if x == 'Mar' else x)
Weather_2013_Sub['Month'] = Weather_2013_Sub['Month'].apply(lambda x: 4 if x == 'Apr' else x)
Weather_2013_Sub['Month'] = Weather_2013_Sub['Month'].apply(lambda x: 5 if x == 'May' else x)
Weather_2013_Sub['Month'] = Weather_2013_Sub['Month'].apply(lambda x: 6 if x == 'Jun' else x)
Weather_2013_Sub['Month'] = Weather_2013_Sub['Month'].apply(lambda x: 7 if x == 'Jul' else x)
Weather_2013_Sub['Month'] = Weather_2013_Sub['Month'].apply(lambda x: 8 if x == 'Aug' else x)
Weather_2013_Sub['Month'] = Weather_2013_Sub['Month'].apply(lambda x: 9 if x == 'Sep' else x)
Weather_2013_Sub['Month'] = Weather_2013_Sub['Month'].apply(lambda x: 10 if x == 'Oct' else x)
Weather_2013_Sub['Month'] = Weather_2013_Sub['Month'].apply(lambda x: 11 if x == 'Nov' else x)
Weather_2013_Sub['Month'] = Weather_2013_Sub['Month'].apply(lambda x: 12 if x == 'Dec' else x)

Weather_2014_Sub['Month'] = Weather_2014_Sub['Month'].apply(lambda x: 1 if x == 'Jan' else x)
Weather_2014_Sub['Month'] = Weather_2014_Sub['Month'].apply(lambda x: 2 if x == 'Feb' else x)
Weather_2014_Sub['Month'] = Weather_2014_Sub['Month'].apply(lambda x: 3 if x == 'Mar' else x)
Weather_2014_Sub['Month'] = Weather_2014_Sub['Month'].apply(lambda x: 4 if x == 'Apr' else x)
Weather_2014_Sub['Month'] = Weather_2014_Sub['Month'].apply(lambda x: 5 if x == 'May' else x)
Weather_2014_Sub['Month'] = Weather_2014_Sub['Month'].apply(lambda x: 6 if x == 'Jun' else x)
Weather_2014_Sub['Month'] = Weather_2014_Sub['Month'].apply(lambda x: 7 if x == 'Jul' else x)
Weather_2014_Sub['Month'] = Weather_2014_Sub['Month'].apply(lambda x: 8 if x == 'Aug' else x)
Weather_2014_Sub['Month'] = Weather_2014_Sub['Month'].apply(lambda x: 9 if x == 'Sep' else x)
Weather_2014_Sub['Month'] = Weather_2014_Sub['Month'].apply(lambda x: 10 if x == 'Oct' else x)
Weather_2014_Sub['Month'] = Weather_2014_Sub['Month'].apply(lambda x: 11 if x == 'Nov' else x)
Weather_2014_Sub['Month'] = Weather_2014_Sub['Month'].apply(lambda x: 12 if x == 'Dec' else x)

Weather_2015_Sub['Month'] = Weather_2015_Sub['Month'].apply(lambda x: 1 if x == 'Jan' else x)
Weather_2015_Sub['Month'] = Weather_2015_Sub['Month'].apply(lambda x: 2 if x == 'Feb' else x)
Weather_2015_Sub['Month'] = Weather_2015_Sub['Month'].apply(lambda x: 3 if x == 'Mar' else x)
Weather_2015_Sub['Month'] = Weather_2015_Sub['Month'].apply(lambda x: 4 if x == 'Apr' else x)
Weather_2015_Sub['Month'] = Weather_2015_Sub['Month'].apply(lambda x: 5 if x == 'May' else x)
Weather_2015_Sub['Month'] = Weather_2015_Sub['Month'].apply(lambda x: 6 if x == 'Jun' else x)
Weather_2015_Sub['Month'] = Weather_2015_Sub['Month'].apply(lambda x: 7 if x == 'Jul' else x)
Weather_2015_Sub['Month'] = Weather_2015_Sub['Month'].apply(lambda x: 8 if x == 'Aug' else x)
Weather_2015_Sub['Month'] = Weather_2015_Sub['Month'].apply(lambda x: 9 if x == 'Sep' else x)
Weather_2015_Sub['Month'] = Weather_2015_Sub['Month'].apply(lambda x: 10 if x == 'Oct' else x)
Weather_2015_Sub['Month'] = Weather_2015_Sub['Month'].apply(lambda x: 11 if x == 'Nov' else x)
Weather_2015_Sub['Month'] = Weather_2015_Sub['Month'].apply(lambda x: 12 if x == 'Dec' else x)


################################################################################################################
#Merge All yearly Weather data into a single Dataframe

Weather = pd.concat([Weather_2009_Sub,Weather_2010_Sub, Weather_2011_Sub, Weather_2012_Sub, Weather_2013_Sub, Weather_2014_Sub,Weather_2015_Sub, Weather_2016_Sub],axis = 0, ignore_index=False)
Weather
print(Weather.dtypes)
print(Weather.shape)

#Read the Train Data and filter for only Women Category
Train = pd.read_csv('Train.csv')
Train.head()
Train = Train[Train.ProductCategory == "WomenClothing"]
Train = Train.reset_index(drop=True).reset_index()
Train.head()

#Merge Train and Weather data with an outer join
Train_W = pd.merge(Weather, Train, on=['Month', 'Year'], how = 'outer')
Train_W.head()

####################################################################################################################
#Holidays Data
Holidays = pd.read_excel('Events_HolidaysData.xlsx', sheetname="Sheet1")
Holidays.head()

Holidays['MonthDate'] = Holidays['MonthDate'].astype(str)
Holidays['MonthDate'].head()

Holidays['Month'] = Holidays['MonthDate'].apply(lambda x: (x.split("-"))[1])
Holidays['Month'] = Holidays['Month'].astype(int)
Holidays['Month'].head()
Holidays.head()

#FinalData['Month'] = FinalData['Month'].astype(str)
#FinalData.dtypes
DropHol = ['Event', 'MonthDate']
Holiday_M = Holidays.drop(DropHol, 1)
Holiday_M.head()

#Dummify DayCategory Variable and perform aggregation based on Year and Month
Holiday_M1 = pd.get_dummies(Holiday_M)
#print(Holiday_M.head())
print(Holiday_M1.head())
print(Holiday_M1.shape)
#Holiday_M1['Year'] = Holiday_M1['Year'].astype(str)
Holiday_Final = Holiday_M1.groupby(['Year','Month']).size().reset_index()
Holiday_Final.head()
Holiday_Final['HolidayInAMonth'] = Holiday_Final[0]
Holiday_Final.columns
Holiday_Final.head()
Holiday_Final = Holiday_Final.drop(0,1)
Holiday_Final.head()

####################################################################################################################
#Read the Macroeconomic Data from Excel
Macro = pd.read_excel('MacroEconomicData.xlsx', sheetname="Sheet1")
Macro.head()

#Split Year and Month into separate columns
Macro['Year'] = Macro['Year-Month'].apply(lambda x: (x.split("-"))[0])
Macro['Year'].head()

Macro['Month'] = Macro['Year-Month'].apply(lambda x: (x.split("-"))[1])
Macro['Month'].head()

#Convert Month into Numerical Data for ease of join with train and weather data
Macro['Month'] = Macro['Month'].apply(lambda x: 1 if x == ' Jan' else x)
Macro['Month'] = Macro['Month'].apply(lambda x: 2 if x == ' Feb' else x)
Macro['Month'] = Macro['Month'].apply(lambda x: 3 if x == ' Mar' else x)
Macro['Month'] = Macro['Month'].apply(lambda x: 4 if x == ' Apr' else x)
Macro['Month'] = Macro['Month'].apply(lambda x: 5 if x == ' May' else x)
Macro['Month'] = Macro['Month'].apply(lambda x: 6 if x == ' Jun' else x)
Macro['Month'] = Macro['Month'].apply(lambda x: 7 if x == ' Jul' else x)
Macro['Month'] = Macro['Month'].apply(lambda x: 8 if x == ' Aug' else x)
Macro['Month'] = Macro['Month'].apply(lambda x: 9 if x == ' Sep' else x)
Macro['Month'] = Macro['Month'].apply(lambda x: 10 if x == ' Oct' else x)
Macro['Month'] = Macro['Month'].apply(lambda x: 11 if x == ' Nov' else x)
Macro['Month'] = Macro['Month'].apply(lambda x: 12 if x == ' Dec' else x)

Macro_M  = Macro[['Month','Year', 'Exports' , 'Monthly Nominal GDP Index (inMillion$)',
                                     'Monthly Real GDP Index (inMillion$)',
                                                                     'CPI',
                                                            'PartyInPower',
                                                       'unemployment rate',
                             'CommercialBankInterestRateonCreditCardPlans',
       'Finance Rate on Personal Loans at Commercial Banks, 24 Month Loan',
                                  'Earnings or wages  in dollars per hour',
                               'AdvertisingExpenses (in Thousand Dollars)',
                          'Cotton Monthly Price - US cents per Pound(lbs)',
                                                             'Change(in%)',
                                   'Average upland planted(million acres)',
                                 'Average upland harvested(million acres)',
                                                  'yieldperharvested acre',
                     'Production (in  480-lb netweright in million bales)',
                      'Mill use  (in  480-lb netweright in million bales)',
]]

Macro_M[u'Year'] = Macro_M[u'Year'].astype(str).astype(int)
Macro_M[u'PartyInPower'] = Macro_M[u'PartyInPower'].astype(str)


Macro_M.PartyInPower.unique()
#As seen above PartyinPower has single distinct value. Thus needs to be dropped

#Combine weather, macroeconomics and train into a single dataframe
Train_ME = pd.merge(Macro_M, Train_W, on=['Month', 'Year'], how = 'outer')
Train_ME

Train_MHT = pd.merge( Train_ME, Holiday_Final , on=['Month', 'Year'], how = 'outer')
Train_MHT.head()

#########################################################################################################################
#Prepare Final Data for Model Building
Drop = ["index", "ProductCategory", "AdvertisingExpenses (in Thousand Dollars)", "PartyInPower"]
FinalData = Train_MHT.drop(Drop, axis = 1)

#Impute missing values
#FinalData['EventOrNot'] = FinalData['EventOrNot'].fillna((FinalData['EventOrNot'].mean()))
#FinalData[u'Temp avg (°C)'] = FinalData[u'Temp avg (°C)'].fillna((FinalData[u'Temp avg (°C)'].mean()))
FinalData['HolidayInAMonth'] = FinalData['HolidayInAMonth'].fillna(0)
print(FinalData.isnull().any())

#Split Test and Train Data
train = FinalData[0:84]
test = FinalData[84:96]


#Y_Train
yt = FinalData[0:84]['Sales(In ThousandDollars)'].fillna(FinalData[0:84]['Sales(In ThousandDollars)'].mean())
print(yt)

columns = [                                                        u'Month',
                                                                    u'Year', 
                                                                 u'Exports',
                                  u'Monthly Nominal GDP Index (inMillion$)',
                                     u'Monthly Real GDP Index (inMillion$)',
                                                                     u'CPI',
                                                       u'unemployment rate',
                             u'CommercialBankInterestRateonCreditCardPlans',
       u'Finance Rate on Personal Loans at Commercial Banks, 24 Month Loan',
                                  u'Earnings or wages  in dollars per hour',
                          u'Cotton Monthly Price - US cents per Pound(lbs)',
                                                             u'Change(in%)',
                                   u'Average upland planted(million acres)',
                                 u'Average upland harvested(million acres)',
                                                  u'yieldperharvested acre',
                     u'Production (in  480-lb netweright in million bales)',
                      u'Mill use  (in  480-lb netweright in million bales)',
                                                               u'EventOrNot',
                                                           u'Temp avg (°C)',
                                                         u'HolidayInAMonth',
                                               u'Sales(In ThousandDollars)']
train = train.reindex(columns=columns)

print(train.shape)
print(test.shape)
#print(Validation.shape)

train['Sales(In ThousandDollars)'] = train['Sales(In ThousandDollars)'].fillna(train['Sales(In ThousandDollars)'].mean())
#train['Sales(In ThousandDollars)']

#####################################|Visualization on the train Data|#####################################################
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
#plt.style.use('ggplot')
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'


#Target variable exploration - 'Sales(In ThousandDollars'
plt.figure(figsize=(12,8))
plt.scatter(range(train.shape[0]), np.sort(train['Sales(In ThousandDollars)'].values))
plt.xlabel('Time(Months)', fontsize=12)
plt.ylabel('Sales(In ThousandDollars)', fontsize=12)
plt.show()

#Variation of Sales with Respect to Months 
plt.figure(figsize=(12,8))
sns.barplot(train['Month'], train['Sales(In ThousandDollars)'], alpha=0.9, color=color[4])
plt.ylabel('Sales', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.title('Variation of Sales with Respect to Months ', fontsize=15 )
#plt.xticks(rotation='vertical')
plt.show()


#Holiday in Month
cnt_srs = train['HolidayInAMonth'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[4])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Number of Holidays in a month', fontsize=12)
plt.show()

#Nominal GDP vs Sales
plt.figure(figsize=(12,8))
plt.scatter(train['Sales(In ThousandDollars)'], np.sort(train['Monthly Nominal GDP Index (inMillion$)'].values), color=color[0])
plt.xlabel('Sales(In ThousandDollars)', fontsize=12)
plt.ylabel('Monthly Nominal GDP Index (inMillion$)', fontsize=12)
plt.title('Nominal GDP Vs Sales', fontsize=15 )
plt.show()

#Highly Correleted Columns
plt.figure(figsize=(12,8))
plt.scatter(train['Monthly Real GDP Index (inMillion$)'], np.sort(train['Monthly Nominal GDP Index (inMillion$)'].values), color=color[0])
plt.xlabel('Monthly Real GDP Index (inMillion$)', fontsize=12)
plt.ylabel('Monthly Nominal GDP Index (inMillion$)', fontsize=12)
plt.title('Real GDP Vs Nominal GDP', fontsize=15 )
plt.show()

train.describe()

#Correlation Matrix
corr = train.corr()
corr = (corr)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.title('Heatmap of Correlation Matrix')
corr

#Holiday vs Sales
sns.boxplot(x="HolidayInAMonth", y="Sales(In ThousandDollars)", data=train)

###########################################################################################################################
#Prepare for Model Building by converting train and test to arrays
#Reservation of Rows for validation resulted in dip in accuracy. Thus none of the data is reserved
#for validation. Train accuracy and Test error is trusted upon. 

Drop = ['Sales(In ThousandDollars)']
train = train.drop(Drop, axis = 1)
#Validation = Validation.drop(Drop, axis = 1)
test = test.drop(Drop, 1)

#Create Arrays
y_train = yt.ravel()
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data
#x_val = Validation.values

#Standardization lead to decrease in accuracy. Thus below code is commented out.
#from sklearn import preprocessing
#x_train = preprocessing.scale(x_train)
#x_test = preprocessing.scale(x_test)



###############################################################################################################
#Model Building
import pandas as pd
import numpy as np
#import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge
 
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 123 # for reproducibility
NFOLDS = 10 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

#Class to extend the Sklearn classifier and VIF defined below
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
        
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

#Check VIF for the train Variables
from statsmodels.stats.outliers_influence import variance_inflation_factor  
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.preprocessing import Imputer 

class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh
        
        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = Imputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped=True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                #print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
        return X
    
transformer = ReduceVIF()
X_VIF = transformer.fit_transform(train, yt)
X_VIF.head()

X_VIF_test = test[['Change(in%)', u'Temp avg (°C)', 'HolidayInAMonth', u'Cotton Monthly Price - US cents per Pound(lbs)']]


######################################################################################################
#Linear Regression

from sklearn.linear_model import LinearRegression
X = x_train
y = y_train 
model = LinearRegression()
model.fit(X, y)
print(model.score(x_train, y_train)) #Train Error: 80.4%

predictions = model.predict(x_test)
print 'Linear Regression Predicted Values'
for i, prediction in enumerate(predictions):
    print 'Predicted: %s' % (prediction)
    
#Try with Variables obtained after VIF
model = LinearRegression()
model.fit(X_VIF, y)
print(model.score(X_VIF, y)) #Train Error: 12.3%

predictions = model.predict(X_VIF_test)
print 'Linear Regression Predicted Values'
for i, prediction in enumerate(predictions):
    print 'Predicted: %s' % (prediction)


#Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
est = GradientBoostingRegressor(n_estimators=70, learning_rate=0.1,max_depth=1 ,random_state=0, loss='ls').fit(X, y)
print(est.score(x_train, y_train)) #Train Error: 94.2

predictions = est.predict(x_test)
for i, prediction in enumerate(predictions):
    print 'Predicted: %s' % (prediction)
    
    
est1 = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01,max_depth=10, random_state=0, loss='ls').fit(X, y)
predictions = est1.predict(x_test)
print(est1.score(x_train, y_train)) #Train Error: 99.9%

for i, prediction in enumerate(predictions):
    print 'Predicted: %s' % (prediction)
    
#Best Model
est3 = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.001,max_depth=5, random_state=0, loss='ls').fit(X, y)
predictions = est3.predict(x_test)
print(est3.score(x_train, y_train)) #Train Error: 86.4%
for i, prediction in enumerate(predictions):
    print 'Predicted: %s' % (prediction)

## use a full grid over all parameters
from sklearn.model_selection import GridSearchCV
clf = GradientBoostingRegressor()
param_grid = {"max_depth": [2,3,4,5],
              "max_features": [1,3,4,5,6,7,8,10],
              "min_samples_split": [20],
              "min_samples_leaf": [20],
              "learning_rate": [0.8, 0.5, 0.3, 0.1, 0.01, 0.001, 0.001, 0.0001]
              }

# run grid search

grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
grid_search.fit(X, y)
grid_search.score(x_train, y_train)
#Train Error: 99.8%

predictions = grid_search.predict(x_test)
for i, prediction in enumerate(predictions):
    print 'Predicted: %s' % (prediction)
    
print("Best parameters set found on development set:")
print()
print(grid_search.best_params_)
print()
print("Grid scores on development set:")
print()
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()
    
########################################################################################################################
#Perceptron Model
from sklearn.linear_model import PassiveAggressiveRegressor
regr = PassiveAggressiveRegressor(random_state=0, C=1.0, average=False, epsilon=0.1,
              fit_intercept=True, loss='epsilon_insensitive',
              max_iter=None, n_iter=None, shuffle=True,
              tol=None, verbose=0, warm_start=False)
regr.fit(X, y)
print(regr.score(x_train, y_train)) #Train Error: 32.86
#PassiveAggressiveRegressor()

predictions = regr.predict(x_test)
for i, prediction in enumerate(predictions):
    print 'Predicted: %s' % (prediction)

############################################################################################################    
#Support Vector Machine Regression
from sklearn import svm
clf1 = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=8, epsilon=0.1, gamma='auto',
    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
clf1.fit(X, y) #66.6%
print(clf1.score(x_train, y_train))



predictions = clf1.predict(x_test)
for i, prediction in enumerate(predictions):
    print 'Predicted: %s' % (prediction)
    
#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_1.fit(X, y)
print(regr_1.score(x_train, y_train)) #Train Error: 88.6%

predictions = regr_1.predict(x_test)
for i, prediction in enumerate(predictions):
    print 'Predicted: %s' % (prediction)
    
#Adaboost Regression
from sklearn.ensemble import AdaBoostRegressor
rng = np.random.RandomState(1)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)
regr_2.fit(X, y)  
print(regr_2.score(x_train, y_train)) #Train: 97.09%
  
predictions = regr_2.predict(x_test)
for i, prediction in enumerate(predictions):
    print 'Predicted: %s' % (prediction)
    
#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regr_RF = RandomForestRegressor(max_depth=5, random_state=0)
regr_RF.fit(X, y)
RandomForestRegressor(bootstrap=True, criterion='mae', max_depth=6,
           max_features= 'auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
           oob_score=False, random_state=0, verbose=0, warm_start=False)
print(regr_RF.feature_importances_)

predictions = regr_RF.predict(x_test)
print(regr_RF.score(x_train, y_train))  #Train: 92.4%
for i, prediction in enumerate(predictions):
    print 'Predicted: %s' % (prediction)
    
#Important Features from Random Forest Model
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(train.columns, regr_RF.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90)

#Important Features from XGBoost
#XGBoost
xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train, y, feature_names=train.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()

#######################################################################################################################
#Stacked Model
#Ridge Regression
ridge_params = {
    'alpha': 0.5
    }

# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 100, 
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 4,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 1
}


# AdaBoost parameters
ada_params = {
    'n_estimators': 10, 
    'learning_rate' : 0.7
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators':  10,
     'max_features':  4,
    'max_depth': 4,
    'learning_rate' : 0.7, 
    'min_samples_leaf': 2,
    'verbose': 1
}

# Support Vector Classifier parameters 
svr_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

# Decision Tree Regressor
dt_params = {
    'max_depth' : 4
    }

rf = SklearnHelper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
gb = SklearnHelper(clf=GradientBoostingRegressor, seed=SEED, params=gb_params)
ada = SklearnHelper(clf=AdaBoostRegressor, seed=SEED, params=ada_params)
ridge = SklearnHelper(clf=Ridge, seed=SEED, params=ridge_params)
dt = SklearnHelper(clf=DecisionTreeRegressor, seed=SEED, params=dt_params)

ridge_oof_train, ridge_oof_test = get_oof(ridge, x_train, y_train, x_test)
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test)
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test)
dt_oof_train, dt_oof_test = get_oof(dt,x_train, y_train, x_test) # Decision Tree Regression

base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'Ridge': ridge_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
     'DecisionTree': dt_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
print(base_predictions_train.head())

x_train_stack = np.concatenate(( rf_oof_train, ada_oof_train, gb_oof_train, dt_oof_train), axis=1)
x_test_stack = np.concatenate(( rf_oof_test, ada_oof_test, gb_oof_test, dt_oof_test), axis=1)

est_stack = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.8,max_depth=4, random_state=0, loss='ls').fit(x_train_stack, y_train)
print(est_stack.score(x_train_stack, y_train)) #Train: 86.48%

predictions = est_stack.predict(x_test_stack)
for i, prediction in enumerate(predictions):
    print 'Predicted: %s' % (prediction)
    

#Feature Importance
rf_feature = rf.feature_importances(x_train,y_train)
gb_feature = gb.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
dt_feature = dt.feature_importances(x_train,y_train)

rf_features = [ 0.19872002,  0.02035471,  0.03680132,  0.07737933,  0.1053616,   0.05174044,
  0.04999982,  0.01923494,  0.02752918,  0.06727491,  0.04697664,  0.02489441,
  0.06589269,  0.01028083,  0.03027401,  0.01294811,  0.01039754,  0.02212421,
  0.0496602,   0.07215507]

dt_features = [  3.00236415e-01,   3.02985764e-02,   0.00000000e+00,   5.96078363e-01,
   3.21002114e-03,   0.00000000e+00,   3.02766604e-02,   0.00000000e+00,
   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.94091256e-02,
   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   2.00474076e-02,
   0.00000000e+00,   0.00000000e+00,   4.43430490e-04,   0.00000000e+00]

ada_features = [ 0.36932797,  0,          0.00227871,  0.08621769,  0.03804946,  0.10489432,
  0.21836761,  0.00674111,  0.04224623,  0.00227994,  0.02376971,  0.00276803,
  0.01416861,  0.00114592,  0.00155959,  0.00042151,  0,          0.02191278,
  0.04339344,  0.02045738]
gb_features = [  3.48531283e-02,   3.24223394e-02,   1.79276565e-02,   6.09773053e-02,
   7.62825046e-02,   1.23603520e-01,   4.15256370e-02,   2.99974336e-02,
   0.00000000e+00,   2.09498906e-02,   2.92687200e-02,   5.47303322e-02,
   5.40183133e-02,   4.73321442e-02,   8.27166871e-02,   3.31762777e-06,
   0.00000000e+00,   1.10950911e-01,   1.01505121e-01,   8.09350383e-02]


#########Feature Importance Visualization#########################################################
#Random Forest Feature Importance
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(train.columns, rf_features):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90)

#decision tree Feature Importance
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(train.columns, dt_features):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90)


#Gradient Boost Feature Importance
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(train.columns, gb_features):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90)

# AdaBoost Feature Importance
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(train.columns, ada_features):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90)