#------------------------------------------------------------------------------------------------------------------
#   Communication test for receiving sensor data from a mobile device
#------------------------------------------------------------------------------------------------------------------
import requests
import time

# IP address and command for the mobile device running Phyphox
IP_ADDRESS = '10.43.104.19'         # Replace with your device's IP address
COMMAND = 'accX&accY&accZ&acc_time'     # Command to fetch acceleration data
BASE_URL = "http://{}/get?{}".format(IP_ADDRESS, COMMAND)

# Function to fetch sensor data from the mobile device
def get_sensor_data():
    try:
        response = requests.get(BASE_URL, timeout=1)
        response.raise_for_status()
        data = response.json()

        accX = data["buffer"]["accX"]["buffer"][0]
        accY = data["buffer"]["accY"]["buffer"][0]
        accZ = data["buffer"]["accZ"]["buffer"][0]
        timestamp = data["buffer"]["acc_time"]["buffer"][0]

        return {
            "time": timestamp,
            "x": accX,
            "y": accY,
            "z": accZ
        }

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Main loop to continuously fetch and print sensor data
samp_rate = 5000
print("Reading real-time data from Phyphox...\nPress Ctrl+C to stop.")
try:
    while True:
        data = get_sensor_data()
        if data and data['time']:
            print(f"t = {data['time']:.3f}s | accX = {data['x']:.4f}, accY = {data['y']:.4f}, accZ = {data['z']:.4f}")        
        time.sleep(1./samp_rate)
except KeyboardInterrupt:
    print("\nReading stopped by user.")

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------