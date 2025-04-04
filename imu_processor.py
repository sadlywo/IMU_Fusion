import numpy as np
from typing import Dict, Optional
from scipy.spatial.transform import Rotation
from dataclasses import dataclass

@dataclass
class FilterParams:
    """Parameters for sensor fusion filters"""
    accel_cutoff: float = 0.1    # Low-pass cutoff frequency for accelerometer (Hz)
    gyro_cutoff: float = 1.0     # High-pass cutoff frequency for gyroscope (Hz)
    mag_cutoff: float = 0.1      # Low-pass cutoff frequency for magnetometer (Hz)
    dt: float = 0.01             # Time step (s)
    alpha: float = 0.98          # Complementary filter gain
    q: np.ndarray = None         # Process noise covariance (Kalman filter)
    r: np.ndarray = None         # Measurement noise covariance (Kalman filter)

class IMUProcessor:
    def __init__(self, params: Optional[FilterParams] = None):
        """
        Initialize IMU data processor
        
        Parameters:
        params: Filter parameters (optional)
        """
        self.params = params if params else FilterParams()
        self.initialize_filters()
        
    def initialize_filters(self):
        """Initialize filter states"""
        # Complementary filter state
        self.orientation = np.zeros(3)  # [roll, pitch, yaw] in radians
        
        # Kalman filter state
        self.x = np.zeros(6)  # State vector: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.P = np.eye(6)    # State covariance matrix
        
        # Initialize noise matrices if not provided
        if self.params.q is None:
            self.params.q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
        if self.params.r is None:
            self.params.r = np.diag([0.1, 0.1, 0.1])
            
    def process_imu_data(self, imu_data: Dict, method: str = 'complementary') -> Dict:
        """
        Process IMU data to compute orientation and trajectory
        
        Parameters:
        imu_data: Dictionary containing raw IMU data
        method: Fusion method ('complementary' or 'kalman')
        
        Returns:
        Dictionary containing processed data (orientation, position, velocity)
        """
        time = imu_data['time']
        accel = imu_data['accel']
        gyro = imu_data['gyro']
        mag = imu_data.get('mag')
        
        # Initialize output arrays
        orientation = np.zeros((len(time), 3))
        position = np.zeros((len(time), 3))
        velocity = np.zeros((len(time), 3))
        
        # Process each time step
        for i in range(len(time)):
            if i == 0:
                dt = time[1] - time[0]
            else:
                dt = time[i] - time[i-1]
                
            if method == 'complementary':
                orientation[i] = self.complementary_filter(
                    accel[i], gyro[i], mag[i] if mag is not None else None, dt)
            else:
                orientation[i] = self.kalman_filter(
                    accel[i], gyro[i], mag[i] if mag is not None else None, dt)
                
            # Compute velocity and position (simple integration)
            if i > 0:
                # Rotate acceleration to world frame
                R = Rotation.from_euler('xyz', orientation[i]).as_matrix()
                world_accel = R @ accel[i] - np.array([0, 0, 9.81])  # Subtract gravity
                
                # Integrate acceleration to get velocity
                velocity[i] = velocity[i-1] + world_accel * dt
                
                # Integrate velocity to get position
                position[i] = position[i-1] + velocity[i] * dt
                
        return {
            'time': time,
            'orientation': orientation,
            'position': position,
            'velocity': velocity
        }
        
    def complementary_filter(self, 
                           accel: np.ndarray,
                           gyro: np.ndarray,
                           mag: Optional[np.ndarray],
                           dt: float) -> np.ndarray:
        """
        Complementary filter for orientation estimation
        
        Parameters:
        accel: Accelerometer measurement (3D vector)
        gyro: Gyroscope measurement (3D vector)
        mag: Magnetometer measurement (3D vector, optional)
        dt: Time step (s)
        
        Returns:
        Orientation angles [roll, pitch, yaw] in radians
        """
        # Normalize accelerometer measurement
        accel_norm = accel / np.linalg.norm(accel)
        
        # Calculate roll and pitch from accelerometer
        roll_acc = np.arctan2(accel_norm[1], accel_norm[2])
        pitch_acc = np.arctan2(-accel_norm[0], 
                              np.sqrt(accel_norm[1]**2 + accel_norm[2]**2))
        
        # Integrate gyroscope measurements
        self.orientation += gyro * dt
        
        # Complementary filter fusion
        self.orientation[0] = self.params.alpha * (self.orientation[0]) + \
                             (1 - self.params.alpha) * roll_acc
        self.orientation[1] = self.params.alpha * (self.orientation[1]) + \
                             (1 - self.params.alpha) * pitch_acc
                             
        # If magnetometer is available, calculate yaw
        if mag is not None:
            # Normalize magnetometer measurement
            mag_norm = mag / np.linalg.norm(mag)
            
            # Calculate yaw from magnetometer
            roll = self.orientation[0]
            pitch = self.orientation[1]
            
            # Rotate magnetometer measurement to horizontal plane
            R_roll = np.array([
                [1, 0, 0],
                [0, np.cos(roll), np.sin(roll)],
                [0, -np.sin(roll), np.cos(roll)]
            ])
            
            R_pitch = np.array([
                [np.cos(pitch), 0, -np.sin(pitch)],
                [0, 1, 0],
                [np.sin(pitch), 0, np.cos(pitch)]
            ])
            
            mag_horiz = R_pitch @ R_roll @ mag_norm
            yaw_mag = np.arctan2(-mag_horiz[1], mag_horiz[0])
            
            # Fuse yaw estimate
            self.orientation[2] = self.params.alpha * (self.orientation[2]) + \
                                 (1 - self.params.alpha) * yaw_mag
        else:
            # Without magnetometer, just use gyro integration for yaw
            pass
            
        return self.orientation
        
    def kalman_filter(self,
                     accel: np.ndarray,
                     gyro: np.ndarray,
                     mag: Optional[np.ndarray],
                     dt: float) -> np.ndarray:
        """
        Kalman filter for orientation estimation
        
        Parameters:
        accel: Accelerometer measurement (3D vector)
        gyro: Gyroscope measurement (3D vector)
        mag: Magnetometer measurement (3D vector, optional)
        dt: Time step (s)
        
        Returns:
        Orientation angles [roll, pitch, yaw] in radians
        """
        # Prediction step
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        
        # State transition
        self.x = F @ self.x
        self.x[3:6] = gyro  # Update angular rates directly from gyro
        
        # Covariance prediction
        self.P = F @ self.P @ F.T + self.params.q
        
        # Measurement update (using accelerometer)
        if np.linalg.norm(accel) > 0:
            accel_norm = accel / np.linalg.norm(accel)
            
            # Calculate roll and pitch from accelerometer
            roll_acc = np.arctan2(accel_norm[1], accel_norm[2])
            pitch_acc = np.arctan2(-accel_norm[0], 
                                  np.sqrt(accel_norm[1]**2 + accel_norm[2]**2))
            
            # Measurement matrix (we only measure roll and pitch from accel)
            H = np.zeros((2, 6))
            H[0, 0] = 1  # Roll
            H[1, 1] = 1  # Pitch
            
            # Measurement
            z = np.array([roll_acc, pitch_acc])
            
            # Kalman gain
            S = H @ self.P @ H.T + self.params.r[0:2, 0:2]
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # State update
            y = z - H @ self.x[0:2]
            self.x += K @ y
            self.P = (np.eye(6) - K @ H) @ self.P
            
        # If magnetometer is available, update yaw
        if mag is not None and np.linalg.norm(mag) > 0:
            mag_norm = mag / np.linalg.norm(mag)
            
            # Calculate yaw from magnetometer
            roll = self.x[0]
            pitch = self.x[1]
            
            # Rotate magnetometer measurement to horizontal plane
            R_roll = np.array([
                [1, 0, 0],
                [0, np.cos(roll), np.sin(roll)],
                [0, -np.sin(roll), np.cos(roll)]
            ])
            
            R_pitch = np.array([
                [np.cos(pitch), 0, -np.sin(pitch)],
                [0, 1, 0],
                [np.sin(pitch), 0, np.cos(pitch)]
            ])
            
            mag_horiz = R_pitch @ R_roll @ mag_norm
            yaw_mag = np.arctan2(-mag_horiz[1], mag_horiz[0])
            
            # Measurement matrix for yaw
            H_yaw = np.zeros((1, 6))
            H_yaw[0, 2] = 1  # Yaw
            
            # Kalman gain for yaw
            S_yaw = H_yaw @ self.P @ H_yaw.T + self.params.r[2, 2]
            K_yaw = self.P @ H_yaw.T / S_yaw
            
            # State update for yaw
            y_yaw = yaw_mag - self.x[2]
            self.x += K_yaw * y_yaw
            self.P = (np.eye(6) - np.outer(K_yaw, H_yaw)) @ self.P
            
        return self.x[0:3]  # Return only orientation angles
