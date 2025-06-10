#------------------------------------------------------------------------------------------------------------------
#   Mobile sensor data acquisition example
#------------------------------------------------------------------------------------------------------------------
import time
import requests
import numpy as np
import threading

import random
import pickle
from datetime import datetime

from scipy.interpolate import interp1d

# Experiment configuration
conditions = [('Jumping Jacks', 1), ('Saludar', 2), ('Saltar cuerda', 3), ('boxeo', 4), ('rotacion tronco', 5), ('Parado', 6)]
n_trials = 10                # Number of trials per condition
n_windows = 10              # Number of windows for each trial

fixation_cross_time = 2     # Time in seconds for attention fixation
preparation_time = 2        # Time in seconds for preparation before each trial
window_time = 1           # Time in seconds for each trial window
rest_time = 1               # Time in seconds for rest between trials

sampling_rate = 20          # Sampling rate in Hz of the output data
max_samp_rate = 5000        # Maximum possible sampling rate
max_window_samples = int(window_time*max_samp_rate)     # Maximum number of samples in each window

trials = n_trials*conditions
random.shuffle(trials)

trial_time = fixation_cross_time + preparation_time + n_windows*window_time + rest_time

# Communication parameters
IP_ADDRESS = '10.43.104.19'
COMMAND = 'accX&accY&accZ&acc_time&gyroX&gyroY&gyroZ'
BASE_URL = "http://{}/get?{}".format(IP_ADDRESS, COMMAND)

# Data buffer
n_signals = 6   # Number of signals (accX, accY, accZ, gyroX, gyroY, gyroZ)
buffer_size = int(2 * len(trials)*trial_time * max_samp_rate)       # Buffer size  (2 times the number of samples in the complete experiment)

buffer = np.zeros((buffer_size, n_signals + 1), dtype='float64')    # Buffer for storing data (channel 0 is time)
buffer_index = 0                                                    # Index for the next data point to be written

# Flag for stopping the data acquisition
stop_recording_flag = threading.Event()

# Mutex for thread-safe access to the buffer
buffer_lock = threading.Lock()

# Function for continuously fetching data from the mobile device
def fetch_data():    
    sleep_time = 1. / max_samp_rate 
    while not stop_recording_flag.is_set():
        try:
            response = requests.get(BASE_URL, timeout=0.5)
            response.raise_for_status()            
            data = response.json()

            global buffer, buffer_index    
            
            with buffer_lock:  # Ensure thread-safe access to the buffer
                buffer[buffer_index:, 0] = data["buffer"]["acc_time"]["buffer"][0]    
                buffer[buffer_index:, 1] = data["buffer"]["accX"]["buffer"][0]
                buffer[buffer_index:, 2] = data["buffer"]["accY"]["buffer"][0]
                buffer[buffer_index:, 3] = data["buffer"]["accZ"]["buffer"][0]
                buffer[buffer_index:, 4] = data["buffer"]["gyroX"]["buffer"][0]
                buffer[buffer_index:, 5] = data["buffer"]["gyroY"]["buffer"][0]
                buffer[buffer_index:, 6] = data["buffer"]["gyroZ"]["buffer"][0]

                buffer_index += 1
            
        except Exception as e:
            print(f"Error fetching data: {e}")

        time.sleep(sleep_time)

# Function for stopping the data acquisition
def stop_recording():
    stop_recording_flag.set()
    recording_thread.join()
    
# Start data acquisition
recording_thread = threading.Thread(target=fetch_data, daemon=True)
recording_thread.start()

# Run experiment
print ("********* Experiment in progress *********")    
time.sleep(fixation_cross_time)  

window_info = []
count  = 0
for t in trials:

    # Fixation cross    
    count = count + 1;
    print ("\n********* Trial {}/{} *********".format(count, len(trials)))    
    time.sleep(fixation_cross_time)    
    
    # Preparation time
    print (t[0])
    time.sleep(preparation_time)

    # Task
    for window in range(n_windows):                
        time.sleep(window_time)
        #with buffer_lock:  # Ensure thread-safe access to the buffer
        window_info.append((t[0], t[1], buffer_index))  

    # Rest time    
    print ("----Rest----")
    time.sleep(rest_time)

# Stop data acquisition
stop_recording()

# Calculate average sampling rate
t = buffer[:buffer_index, 0]    # Time data
diff_t = np.diff(t)             # Time differences

print("Min sampling rate: {:.2f} Hz".format(1. / np.max(diff_t)))
print("Max sampling rate: {:.2f} Hz".format(1. / np.min(diff_t)))
print("Average sampling rate: {:.2f} Hz".format(1. / np.mean(diff_t)))

# Interpolation functions for uniform sampling
interp_x1 = interp1d(t, buffer[:buffer_index, 1], kind='linear', fill_value="extrapolate") # Interpolation function for acceleration X
interp_x2 = interp1d(t, buffer[:buffer_index, 2], kind='linear', fill_value="extrapolate") # Interpolation function for acceleration Y
interp_x3 = interp1d(t, buffer[:buffer_index, 3], kind='linear', fill_value="extrapolate") # Interpolation function for acceleration Z
interp_g1 = interp1d(t, buffer[:buffer_index, 4], kind='linear', fill_value="extrapolate")
interp_g2 = interp1d(t, buffer[:buffer_index, 5], kind='linear', fill_value="extrapolate")
interp_g3 = interp1d(t, buffer[:buffer_index, 6], kind='linear', fill_value="extrapolate")

# Separate the data for each trial
window_samples = int(sampling_rate * window_time)  # Number of samples in each window
data = []
for w in window_info:
    condition = w[0]        # Condition name
    condition_id = w[1]     # Condition ID
    start_index =  w[2]     # Start index of the window in the buffer

    # Calculate the uniform time vector for the window
    t_start = buffer[start_index, 0]    # Start time of the window
    t_uniform = np.linspace(t_start, t_start + window_time, int(window_time * sampling_rate))    

    # Interpolate the signals for the uniform time vector
    signal_data = np.column_stack((
    interp_x1(t_uniform), interp_x2(t_uniform), interp_x3(t_uniform),
    interp_g1(t_uniform), interp_g2(t_uniform), interp_g3(t_uniform)
))   
    
    # Append the data for this window
    data.append((condition, condition_id, signal_data))

# Save data
nombre = "Augusto"  
outputFile = open(f"{nombre}_data.obj", 'wb')
pickle.dump(data, outputFile)
outputFile.close()

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------