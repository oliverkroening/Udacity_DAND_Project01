from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt

def mov_avg(data,N):
    ma_val = np.array([])
    for i in range((N+1)/2-1,data.size-(N-1)/2,1):
        window = data[i-(N-1)/2:i+(N-1)/2]
        avgval = np.mean(window)
        ma_val = np.append(ma_val,avgval)
    return ma_val

# Load and Prepare Data
glob_data = np.loadtxt(open("global_avg_temp.csv", "rb"),delimiter=",",skiprows=1)
glob_year = glob_data[:,0]
glob_temp = glob_data[:,1]
glob_min_year = min(glob_year)
glob_max_year = max(glob_year)
loc_data = np.loadtxt(open("hamburg_avg_temp.csv", "rb"),delimiter=",",skiprows=1)
loc_year = loc_data[:,0]
loc_temp = loc_data[:,1]
loc_min_year = min(loc_year)
loc_max_year = max(loc_year)
min_year = max([glob_min_year,loc_min_year])
max_year = min([glob_max_year,loc_max_year])
year = np.arange(min_year,max_year,1)
glob_temp = glob_temp[np.where(glob_year==min_year)[0]:np.where(glob_year==max_year)[0]]
loc_temp = loc_temp[np.where(loc_year==min_year)[0]:np.where(loc_year==max_year)[0]]

# Moving Average
N = 7
glob_temp_ma = mov_avg(glob_temp,N)
loc_temp_ma = mov_avg(loc_temp,N)
year_ma = year[(N+1)/2-1:year.size-(N-1)/2]

# Calculate Linear Regression
glob_linRegVals = np.polyfit(year_ma,glob_temp_ma,1)
loc_linRegVals = np.polyfit(year_ma,loc_temp_ma,1)
glob_LR_temp = glob_linRegVals[0]*year_ma + glob_linRegVals[1]
loc_LR_temp = loc_linRegVals[0]*year_ma + loc_linRegVals[1]

# Plot Noisy Data
fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.plot(year,loc_temp,'b',label='Hamburg')
plt.plot(year,glob_temp,'r',label='Global')
plt.ylabel('Average Temperature [Degree Celsius]',fontname="Times")
plt.xlabel('Time [Year]',fontname="Times")
plt.xlim(min_year,max_year)
plt.grid(True)
ax.legend(loc='lower right')
plt.title('Unprocessed yearly average temperature')
plt.savefig('WeatherTrends_noisy.eps', format='eps', dpi=1000)

# Plot Filtered Data
fig = plt.figure(2)
ax = fig.add_subplot(111)
plt.plot(year_ma,loc_temp_ma,'b',label='Hamburg')
plt.plot(year_ma,glob_temp_ma,'r',label='Global')
plt.ylabel('Average Temperature [Degree Celsius]',fontname="Times")
plt.xlabel('Time [Year]',fontname="Times")
plt.xlim(year_ma[0],year_ma[year_ma.size-1])
plt.grid(True)
ax.legend(loc='lower right')
plt.title('7-year moving average of temperature data')
plt.savefig('WeatherTrends_filtered.eps', format='eps', dpi=1000)

# Plot Linear Regression
fig = plt.figure(3)
ax = fig.add_subplot(111)
plt.plot(year_ma,loc_temp_ma,'b.',label='Hamburg data points')
plt.plot(year_ma,glob_temp_ma,'r.',label='Global data points')
plt.plot(year_ma,loc_LR_temp,'b',label='Hamburg linear regression')
plt.plot(year_ma,glob_LR_temp,'r',label='Global linear regression')
plt.ylabel('Average Temperature [Degree Celsius]',fontname="Times")
plt.xlabel('Time [Year]',fontname="Times")
plt.xlim(year_ma[0],year_ma[year_ma.size-1])
plt.grid(True)
ax.legend(loc='lower right')
plt.title('Linear regression models of temperature data (1950 - 2013)')
plt.savefig('WeatherTrends_LinReg.eps', format='eps', dpi=1000)

print 'LinReg global: T='+str(glob_linRegVals[0])+'*t+'+str(glob_linRegVals[1])
print 'LinReg local: T='+str(loc_linRegVals[0])+'*t+'+str(loc_linRegVals[1])
plt.show()
