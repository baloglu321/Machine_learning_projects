import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from utils import draw_graph


km_list=draw_graph()
k = pd.concat(km_list, axis=1).T[['clusters','inertia']]

# Visualize
fig, ax = plt.subplots(figsize =(12, 8))
fig.patch.set_facecolor('white')
mpl.rcParams['font.family'] = 'Ubuntu'
mpl.rcParams['font.size'] = 14

plt.plot(k['clusters'], k['inertia'], 'bo-', color = '#00538F')

# Remove ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Remove axes splines
for i in ['top','right']:
    ax.spines[i].set_visible(False)

ax.set_xticks(range(0,21,2))
ax.set(xlabel='Cluster', ylabel='Inertia');

plt.suptitle('The Elbow Method: Optimal Number of Clusters', size=26);

st.pyplot(fig)