import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


# Data from kaggle
data = pd.read_csv(r'C:\Users\Usuario\Desktop\PYTHON\Indicador BTC\data.csv').dropna()

# Group data into weekly intervals
data['Date'] = pd.to_datetime(data['Timestamp'], unit='s')  
data.set_index('Date', inplace=True)
data_wk = data.resample('W').last()
data_kg = data_wk[['Open','Close','High','Low']]

# Get the most recent data with yfinance
start = '2021-04-13'
data_yf = yf.download('BTC-USD', start = start, interval = '1wk', progress = False )
data_yf = data_yf[['Open','Close','High','Low']]

# Concatenate both datsets
btc = pd.concat([data_kg,data_yf], axis = 0)

# Compute 20 week SMA
btc['SMA 20 Week'] = pd.Series(btc['Close']).rolling(20).mean()
btc = btc.dropna()

# Plot dataset
px.line(btc, y = [btc['SMA 20 Week'],btc['Close']], log_y=True, title = 'BTC Price History')

# Extension from 20 week SMA:
btc['SMA Extension'] = btc['Close']/btc['SMA 20 Week']
px.line(btc, y = btc['SMA Extension'], title = 'Extension from 20 week SMA')

# Logarithmic extension:
btc['Log Extension'] = np.log(abs(btc['SMA Extension']))
px.line(btc, y = btc['Log Extension'], title = 'Logarithmic extension from 20 week SMA')

# Logarithmic extension (only positive values):
btc['Log Extension (positive)'] = btc['Log Extension']*(btc['Log Extension']>0)
px.line(btc, y = btc['Log Extension (positive)'], title = 'Logarithmic extension from 20 week SMA (positive values)')

# Select the 3 highest peaks:
y = btc['Log Extension (positive)'].values
x = np.arange(0,len(y))
peaks, _ = find_peaks(y, height = 1, distance = 20)
y_data = y[peaks]; x_data = x[peaks]

# Define the exponential function:
def exponential_func(x, a, b):
    return a * np.exp(-b * x)

# Provide initial guesses for parameters a and b:
initial_guesses = [1.0, 0.01]

# Fit the exponential curve to the data with initial guesses:
params, covariance = curve_fit(exponential_func, x_data, y_data, p0=initial_guesses)
a_fit, b_fit = params
btc['Fit'] = exponential_func(x, a_fit, b_fit)

fig = px.line(btc, y = [btc['Log Extension (positive)'], btc['Fit']], title = 'Logarithmic extension fitted')
fig.update_layout(showlegend=False)

# Oscillator:
btc['Oscillator'] = btc['Log Extension (positive)']/btc['Fit']
btc['Line'] = 1
fig = px.line(btc, y = [btc['Oscillator'],btc['Line']], title = 'Bitcoin Selling Indicator')
fig.update_layout(showlegend=False)

# Selling Band:
btc['Band'] = np.exp(btc['Fit'])*btc['SMA 20 Week']
fig = px.line(btc, y = [btc['Band'],btc['Close']], title = 'Bitcoin Selling Indicator', log_y= True)
fig.update_layout(showlegend=False)
fig.show()