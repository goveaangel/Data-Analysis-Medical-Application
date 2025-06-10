#------------------------------------------------------------------------------------------------------------------
#   Mobile sensor data acquisition and processing
#------------------------------------------------------------------------------------------------------------------
import pickle
import numpy as np
from scipy import stats

# Load data
file_name = '/Users/angelgovea/Library/CloudStorage/OneDrive-InstitutoTecnologicoydeEstudiosSuperioresdeMonterrey/4to Semestre/Modelacion del Aprendizaje con Inteligencia Artificial/proyecto1/data/Augusto_data.obj'
inputFile = open(file_name, 'rb')
experiment_data = pickle.load(inputFile)

# Process each trial and build data matrices
features = []
for tr in experiment_data:
    
    label = tr[1]       # Activity ID (e.g., 0, 1, ..., 5)
    signal = tr[2]      # Signal shape: (samples, 6 axes: accX, accY, accZ, gyroX, gyroY, gyroZ)

    feat = [label]      # Start the feature list with the activity label

    rms = 0
    for s in range(6):
        sig = signal[:, s]

        feat.append(np.average(sig))
        feat.append(np.std(sig))
        feat.append(stats.kurtosis(sig))
        feat.append(stats.skew(sig))
        rms += np.sum(sig**2)

        feat.append(np.median(sig))
        feat.append(np.max(sig))
        feat.append(np.min(sig))      

        feat.append(np.ptp(sig))  # Peak-to-peak
        q75, q25 = np.percentile(sig, [75 ,25])
        feat.append(q75 - q25)  # Interquartile Range (IQR)
        zero_crossings = ((sig[:-1] * sig[1:]) < 0).sum()
        feat.append(zero_crossings)  # Zero Crossing Rate
        energy = np.sum(sig**2)
        feat.append(energy)  # Energy   
            
    acc_mag = np.linalg.norm(signal[:, 0:3], axis=1)  # Magnitude of accX, accY, accZ
    gyro_mag = np.linalg.norm(signal[:, 3:6], axis=1)  # Magnitude of gyroX, gyroY, gyroZ

    feat.append(np.mean(acc_mag))        # Mean acc magnitude
    feat.append(np.std(acc_mag))         # Std acc magnitude
    feat.append(np.sum(acc_mag ** 2) / len(acc_mag))   # Energy acc magnitude

    feat.append(np.mean(gyro_mag))       # Mean gyro magnitude
    feat.append(np.std(gyro_mag))        # Std gyro magnitude
    feat.append(np.sum(gyro_mag ** 2) / len(gyro_mag)) # Energy gyro magnitude

    rms = np.sqrt(rms)

    feat.append(rms)
    features.append(feat)      

# Build x and y arrays
processed_data =  np.array(features)
x = processed_data[:,1:]
y = processed_data[:,0]

# Save processed data
np.savetxt("activity_data_Augusto.txt", processed_data)

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------