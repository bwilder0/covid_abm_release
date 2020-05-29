import json 
import pandas as pd

df = pd.read_csv('italy_region_data.csv')

d = json.load(open('translation.json','r',encoding='latin-1'))

renames = {}
for item in d:
	renames[item['Nome campo']] = item['Field name']

df = df.rename(columns=renames)

df.to_csv('italy_region_data_english.csv',index=False)

df = df[df['region']=='Lombardia']

print(df.head())

df.to_csv('lombardy_data_english.csv',index=False)

df = df[['date','dead']]
df = df.rename(columns={'date':'Date','dead':'Deaths'})
# print(df['Date'])
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.strftime('%m/%d/%y %H:%M')

df.to_csv('lombardy_data_deaths.csv',index=False)

