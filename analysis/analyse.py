from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis'
def read(name): return pd.read_csv(ROOT/name)
w=read('Withings/weight.csv')
w['date']=pd.to_datetime(w['Date'])
w=w.rename(columns={'Weight (kg)':'weight','Fat mass (kg)':'fat','Muscle mass (kg)':'muscle','Hydration (kg)':'water'})
w['bf']=100*w.fat/w.weight
h=read('Hevy/workouts.csv')
h['start']=pd.to_datetime(h.start_time,format='%b %d, %Y, %I:%M %p')
h['end']=pd.to_datetime(h.end_time,format='%b %d, %Y, %I:%M %p')
h['volume']=h.weight_kg*h.reps
sessions=h.groupby('start').agg(title=('title','first'),end=('end','first'),sets=('set_type','size'),volume=('volume','sum'))
sessions['minutes']=(sessions.end-sessions.index).dt.total_seconds()/60
sessions['sessions']=1
s=read('Withings/sleep.csv')
for col in ['from','to']: s[col]=pd.to_datetime(s[col],utc=True).dt.tz_convert('Australia/Perth').dt.tz_localize(None)
s['date']=s['to'].dt.normalize()
s['sleep_hours']=s[['light (s)','deep (s)','rem (s)']].sum(axis=1,min_count=1)/3600
s['awake_hours']=s['awake (s)']/3600
s['sleep_hr']=s['Average heart rate'].replace(0,np.nan)
# Longest episode per waking date avoids counting overlapping device recordings twice.
sn=s.sort_values('sleep_hours').drop_duplicates('date',keep='last').set_index('date')
st=read('Withings/aggregates_steps.csv'); st['date']=pd.to_datetime(st.date)
st=st.groupby('date').value.max().rename('steps')
a=read('Withings/activities.csv')
a['date']=pd.to_datetime(a['from'],utc=True).dt.tz_convert('Australia/Perth').dt.tz_localize(None)
a['minutes']=(pd.to_datetime(a['to'],utc=True)-pd.to_datetime(a['from'],utc=True)).dt.total_seconds()/60
print('COVERAGE')
for name,df,dates in [('weight',w,w.date),('Hevy sets',h,h.start),('sleep',s,s.date),('steps',st,st.index),('activities',a,a.date)]:
 print(name,len(df),min(dates),max(dates))
print('Weight nulls/ranges',w[['weight','fat','bf','muscle','water']].describe().round(2).to_string())
print('Hevy sessions',len(sessions),'titles',sessions.title.value_counts().to_dict(),'types',h.set_type.value_counts().to_dict(),'duration',sessions.minutes.describe().to_dict())
print('Sleep duplicates',s.date.duplicated().sum(),'activity types',a['Activity type'].value_counts().to_dict())
print('weight years',w.groupby(w.date.dt.year).size().to_dict(),'steps years',st.groupby(st.index.year).agg(['count','mean']).round(1).to_dict())
wd=w.groupby(w.date.dt.normalize())[['weight','fat','bf','muscle','water']].median()
d=wd.reindex(pd.date_range(min(wd.index.min(),st.index.min()),max(wd.index.max(),st.index.max()),freq='D'))
d=d.join(st).join(sn[['sleep_hours','sleep_hr','awake_hours']])
hd=sessions.groupby(sessions.index.normalize())[['sessions','sets','volume','minutes']].sum()
d=d.join(hd)
hc=(d.index>=sessions.index.min().normalize())&(d.index<=sessions.index.max().normalize())
d.loc[hc,['sessions','sets','volume','minutes']]=d.loc[hc,['sessions','sets','volume','minutes']].fillna(0)
weekly=d.resample('W-SUN').mean()
for c in ['sessions','sets','volume','minutes']: weekly[c]=d[c].resample('W-SUN').sum(min_count=7)
weekly['weigh_days']=d.fat.resample('W-SUN').count()
weekly['step_days']=d.steps.resample('W-SUN').count()
weekly['sleep_days']=d.sleep_hours.resample('W-SUN').count()
weekly.loc[weekly.weigh_days<2,['weight','fat','bf','muscle','water']]=np.nan
weekly.loc[weekly.step_days<5,'steps']=np.nan
weekly.loc[weekly.sleep_days<4,['sleep_hours','sleep_hr','awake_hours']]=np.nan
for c in ['fat','weight','bf']:
 for lag in [1,2,4]: weekly[f'{c}_change_{lag}w']=weekly[c].shift(-lag)-weekly[c]
monthly=d.resample('MS').mean()
monthly['weigh_days']=d.fat.resample('MS').count()
monthly['sessions']=d.sessions.resample('MS').sum(min_count=1)
monthly['sets']=d.sets.resample('MS').sum(min_count=1)
print('MONTHLY\n',monthly.round(2).to_string())
print('CORRELATIONS (Pearson, rank)')
for period,df in [('all',weekly),('Hevy overlap',weekly.loc[sessions.index.min():])]:
 for x in ['steps','sleep_hours','sleep_hr','awake_hours','sessions','sets','volume','minutes']:
  for lag in [1,4]:
   z=df[[x,f'fat_change_{lag}w']].dropna()
   if len(z)>8: print(period,x,lag,len(z),round(z.corr().iloc[0,1],3),round(z.corr(method='spearman').iloc[0,1],3))
d.to_csv(OUT/'daily.csv'); weekly.to_csv(OUT/'weekly.csv'); monthly.to_csv(OUT/'monthly.csv'); sessions.to_csv(OUT/'sessions.csv')
print('MEASUREMENT QUALITY')
print('weight unique days',len(wd),'fat days',wd.fat.count(),'same day repeats',len(w)-len(wd),'morning percent',round(100*(w.date.dt.hour<12).mean(),1))
adj=wd.diff(); adj=adj.loc[wd.index.to_series().diff().dt.days<=3]
print('Short gap absolute fat differences',adj.fat.abs().quantile([.5,.9,.95]).to_dict(),'water-fat change corr',adj[['fat','water']].corr().iloc[0,1])
print('sleep episode hours',s.sleep_hours.describe().to_dict())
print('steps extremes',st.nlargest(8).to_dict(),'zeros',(st==0).sum())
print('Monthly step coverage',d.steps.resample('MS').count().tail(22).to_dict())
print('MONTHLY CHANGE CORRELATIONS')
for col in ['fat','weight','bf','muscle']: monthly[col+'_change']=monthly[col].diff()
for period,df in [('2023+',monthly.loc['2023-01-01':]),('2023-25',monthly.loc['2023-01-01':'2025-12-01']),('2026',monthly.loc['2026-01-01':])]:
 for x in ['steps','sleep_hours','sessions','sets']:
  z=df[[x,'fat_change','weight_change','muscle_change']].dropna()
  print(period,x,len(z),z.corr().loc[x].round(3).to_dict())
# Fixed, nonoverlapping 28-day blocks: exposure over block; change between
# median first-week and last-week measurements. Require >=2 fat days at each end.
blocks=[]
for start in pd.date_range('2023-01-02',d.index.max()-pd.Timedelta(days=27),freq='28D'):
 end=start+pd.Timedelta(days=27); b=d.loc[start:end]; b0=b.iloc[:7]; b1=b.iloc[-7:]
 if b0.fat.count()<2 or b1.fat.count()<2: continue
 row={'start':start,'end':end,'fat_change':b1.fat.median()-b0.fat.median(),'weight_change':b1.weight.median()-b0.weight.median(),'bf_change':b1.bf.median()-b0.bf.median()}
 for x in ['steps','sleep_hours','sleep_hr']: row[x]=b[x].mean() if b[x].count()>=20 else np.nan
 for x in ['sessions','sets','minutes','volume']:row[x]=b[x].sum() if b[x].count()==28 else np.nan
 blocks.append(row)
blocks=pd.DataFrame(blocks)
print('28 DAY BLOCKS',len(blocks))
print(blocks.round(2).to_string(index=False))
print('28 DAY CORRELATIONS',blocks.select_dtypes('number').corr().fat_change.round(3).to_dict())
blocks.to_csv(OUT/'blocks.csv',index=False)
# Source overlap: weight-training activities absent from Hevy are not assumed inactivity.
wa=a[a['Activity type']=='Weights'].copy()
wa['hevy_day']=wa.date.dt.normalize().isin(sessions.index.normalize())
print('Withings weights matched Hevy date',wa.hevy_day.value_counts().to_dict())
print('No Hevy matched Withings weights by year',wa[~wa.hevy_day].groupby(wa.date.dt.year).size().to_dict())
print('Most common exercises',h.exercise_title.value_counts().head(12).to_dict())
print('SENSITIVITY')
for label,df in [('all blocks',blocks),('2023-25 blocks',blocks[blocks.start.dt.year<2026])]:
 for x in ['steps','sessions','sets','sleep_hours']:
  z=df[[x,'fat_change']].dropna(); rng=np.random.default_rng(42); rs=[]
  for _ in range(2000):
   sample=z.iloc[rng.integers(0,len(z),len(z))]
   rs.append(sample.corr().iloc[0,1])
  print(label,x,len(z),'r',round(z.corr().iloc[0,1],3),'descriptive bootstrap CI',np.round(np.nanquantile(rs,[.025,.975]),3))
print('RECENT PHASES')
for start,end in [('2025-10-01','2026-03-31'),('2026-04-01','2026-09-26'),('2026-02-01','2026-05-31'),('2026-06-01','2026-09-26')]:
 sub=d.loc[start:end]; print(start,end,'sessions',sub.sessions.sum(),'steps',sub.steps.mean(),'sleep',sub.sleep_hours.mean())
# Compare fat and weight using exactly the same measurement days at key endpoints.
print('PAIRED MONTH ENDPOINTS')
paired=wd.dropna(subset=['fat','weight'])
for date in ['2023-01','2023-11','2024-01','2024-12','2025-10','2026-03','2026-05','2026-09']:
 print(date,paired.loc[date].mean().round(3).to_dict(), 'n',len(paired.loc[date]))
print('SLEEP UNDER SIX HOURS',sn.groupby(sn.index.year).sleep_hours.apply(lambda x:(x<6).mean()).round(3).to_dict())
print('HEVY ANNUAL',sessions.groupby(sessions.index.year).agg(sessions=('sessions','sum'),minutes=('minutes','median'),sets=('sets','median')).to_dict())
