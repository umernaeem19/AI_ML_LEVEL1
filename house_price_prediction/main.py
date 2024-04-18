import pandas as pd
import numpy as np
from matplotlib import pyplot as plt    
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)

df1=pd.read_csv('C:/Users/hp/Desktop/AI_ML/house_price_prediction/data.csv')
df1.shape
df1.groupby('area_type')['area_type'].agg('count')
#print(df1)