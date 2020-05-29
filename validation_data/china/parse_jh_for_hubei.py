import pandas as pd 
import numpy as np 

df = pd.read_csv('time_series_covid19_deaths_global.csv')
print(df.head())
cols = df.columns.values.tolist()

drop_cols = ['Country/Region','Lat','Long']
for col in drop_cols:
	cols.remove(col)

df = df[cols]

print(df.columns.values)

df = df[df['Province/State']=='Hubei']
print(df)



deaths = df.values.reshape(-1)[1:]
dates = np.array(cols[1:])


data = np.concatenate([dates.reshape(-1,1), deaths.reshape(-1,1)],axis=1)


df_out = pd.DataFrame(data,columns=['Date','Deaths'])

df_out.to_csv('hubei.csv',index=False)
