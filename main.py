import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import calendar
from plotly import graph_objs as go
import plotly.express as px
import plotly.offline as py

import plotly.io as pio
pio.renderers = "browser"

df = pd.read_csv("hotel_bookings.csv")

df = df.dropna(axis='index')

cekNA = df.isna().sum()

cekNull = df.isnull().sum()
month_value = df['arrival_date_month'].unique()

d = {'January': 1,
     'February': 2,
     'March': 3,
     'April': 4,
     'May': 5,
     'June': 6,
     'July': 7,
     'August:': 8,
     'September': 9,
     'October': 10,
     'November': 11,
     'December': 12}

df['month'] = df['arrival_date_month'].map(d)
df['month'] = df['month'].fillna(8)
df['month'] = df['month'].dropna(axis='index')
dfmonthcek = df['month'].isna().sum()
columnlist = df.columns.values
# pd.to_datetime(df.Y*10000+df.M*100+df.D,format='%Y%m%d')

df['date'] = pd.to_datetime(df.arrival_date_year * 10000 + df.month * 100 + df.arrival_date_day_of_month,
                            format='%Y%m%d')

df2 = df.drop(['is_canceled', 'lead_time', 'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number',
               'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
               'distribution_channel', 'previous_cancellations', 'previous_bookings_not_canceled',
               'reserved_room_type', 'booking_changes', 'deposit_type', 'agent',
               'company', 'days_in_waiting_list', 'adr', 'required_car_parking_spaces', 'total_of_special_requests',
               'reservation_status', 'reservation_status_date', 'month', 'meal'], axis='columns')

dummy1 = pd.get_dummies(df2.hotel)
dummy1 = dummy1.drop(['City Hotel'], axis='columns')

dummy2 = pd.get_dummies(df2.country)
dummy2 = dummy2.drop('USA', axis='columns')

dummy3 = pd.get_dummies(df2.market_segment)
dummy3 = dummy3.drop('Online TA', axis='columns')

dummy4 = pd.get_dummies(df2.customer_type)
dummy4 = dummy4.drop('Transient-Party', axis='columns')

e = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}

df2['assigned_room_type'] = df2['assigned_room_type'].map(e)

df3 = df2.drop(['hotel', 'country', 'market_segment', 'assigned_room_type', 'customer_type', 'date'], axis='columns')

final = pd.concat([df3, dummy4], axis='columns')

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3)

y_predict = km.fit_predict(final)


final['cluster'] = y_predict
final['index'] = range(1, len(final) + 1)

cektrace0 = final.loc[final['cluster'] == 0]
cektrace1 = final.loc[final['cluster'] == 1]
cektrace2 = final.loc[final['cluster'] == 2]


"""plt.scatter(final['index'],final['cluster'],c='b')
plt.show()"""

scatter = px.scatter(data_frame=final,x='index',
                     y='cluster',
                     color='cluster')
pio.show(scatter)
py.plot(scatter)
































"""from sklearn.decomposition import PCA
cols = final.columns[1:]
pca = PCA(n_components=2)
final['x'] = pca.fit_transform(final[cols])[:,0]
final['y'] = pca.fit_transform(final[cols])[:,1]

trace0 = go.Scatter(x=final[final.cluster == 0]['x'],
                    y=final[final.cluster == 0]['y'],
                    name='Cluster 0',
                    mode='markers',
                    marker=dict(size=10,
                                color="aqua",
                                line=dict(width=1,color="rgb(0,0,0)")))

trace1 = go.Scatter(x=final[final.cluster == 1]['x'],
                    y=final[final.cluster == 1]['y'],
                    name='Cluster 0',
                    mode='markers',
                    marker=dict(size=10,
                                color="green",
                                line=dict(width=1,color="rgb(0,0,0)")))
trace2 = go.Scatter(x=final[final.cluster == 2]['x'],
                    y=final[final.cluster == 2]['y'],
                    name='Cluster 0',
                    mode='markers',
                    marker=dict(size=10,
                                color="orange",
                                line=dict(width=1,color="rgb(0,0,0)")))

data = [trace0,trace1,trace2]

graphy = py.plot(data)"""""