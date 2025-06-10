#------------------------------------------------------------------------------------------------------------------
#   Real-time plot for acceleration data.
#------------------------------------------------------------------------------------------------------------------
import time
import requests
from collections import deque
from threading import Thread

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Communication parameters
IP_ADDRESS = '10.43.104.19'
COMMAND = 'accX&accY&accZ&acc_time'
BASE_URL = "http://{}/get?{}".format(IP_ADDRESS, COMMAND)

# Data acquisition parameters
samps_per_frame = 200           # Number of samples per frame
sampling_rate = 20              # Hz (Sampling rate)
sleep_time = 1. / sampling_rate # Sleep time in seconds for each request

# Data buffers (circular buffers)
t = deque(maxlen = samps_per_frame)
x1 = deque(maxlen = samps_per_frame)
x2 = deque(maxlen = samps_per_frame)
x3 = deque(maxlen = samps_per_frame)

# Function for continuously fetching data from the mobile device
def fetch_data():
    
    acquire = True
    
    while acquire:
        try:
            response = requests.get(BASE_URL, timeout=1)
            response.raise_for_status()
            data = response.json()
            accX = data["buffer"]["accX"]["buffer"][0]
            accY = data["buffer"]["accY"]["buffer"][0]
            accZ = data["buffer"]["accZ"]["buffer"][0]
            timestamp = data["buffer"]["acc_time"]["buffer"][0]

            t.append(timestamp)
            x1.append(accX)
            x2.append(accY)
            x3.append(accZ)           

        except Exception as e:
            print(f"Error: {e}")
            acquire = False

        time.sleep(sleep_time)

# Initialize plots
fig, (ax1, ax2, ax3) = plt.subplots(3)

# 
def animate(i):
    if len(t) > 1:
        ax1.clear()
        ax1.plot(t, x1)
        ax1.set_ylim(min(x1) - 1, max(x1) + 1)
        ax1.set_ylabel("X")

        ax2.clear()
        ax2.plot(t, x2)
        ax1.set_ylim(min(x2) - 1, max(x2) + 1)
        ax2.set_ylabel("Y")

        ax3.clear()
        ax3.plot(t, x3)
        ax1.set_ylim(min(x3) - 1, max(x3) + 1)
        ax3.set_ylabel("Z")
        ax3.set_xlabel("Time (s)")

# Launch data fetching in a separate thread
thread = Thread(target=fetch_data, daemon=True)
thread.start()

# Set up the animation
ani = animation.FuncAnimation(fig, animate, interval=200, cache_frame_data=False)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------