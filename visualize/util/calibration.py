import numpy as np


class CalibrationData:
    def __init__(self, calib_file):
        calibs = read_calib_data(calib_file)
        P = calibs['P2']
        
        self.P = np.reshape(P, (3, 4))
        self.Tr_lidar_to_cam = np.reshape(calibs["Tr_velo_to_cam"], (3,4))
        R0 = calibs["R0_rect"]
        self.R0 = np.reshape(R0, (3, 3))


def read_calib_data(filepath):
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key,value = line.split(':')
                data[key] = np.array([float(x) for x in value.split()])
            except:
                pass
          
    return data

