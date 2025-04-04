import numpy as np 
import time
from imu_processor import IMUProcessor
from visualization import IMUVisualizer
from data_io import IMUDataIO

def generate_simulated_data(duration=10, freq=100, imu_type='6dof'):
    """Generate simulated IMU data for testing"""
    n_samples = int(duration * freq)
    time_vals = np.linspace(0, duration, n_samples)
    
    # Generate accelerometer data (m/s²)
    accel = np.zeros((n_samples, 3))
    accel[:, 0] = 0.1 * np.sin(2 * np.pi * 0.5 * time_vals)  # X-axis
    accel[:, 1] = 0.2 * np.sin(2 * np.pi * 0.3 * time_vals)  # Y-axis 
    accel[:, 2] = 9.81 + 0.1 * np.random.randn(n_samples)    # Z-axis (gravity + noise)
    
    # Generate gyroscope data (rad/s)
    gyro = np.zeros((n_samples, 3))
    gyro[:, 0] = 0.5 * np.sin(2 * np.pi * 0.2 * time_vals)  # Roll rate
    gyro[:, 1] = 0.3 * np.sin(2 * np.pi * 0.4 * time_vals)  # Pitch rate
    gyro[:, 2] = 0.1 * np.sin(2 * np.pi * 0.1 * time_vals)  # Yaw rate
    
    # Generate magnetometer data (μT) if 9DOF
    mag = None
    if imu_type == '9dof':
        mag = np.zeros((n_samples, 3))
        mag[:, 0] = 50 + 5 * np.sin(2 * np.pi * 0.1 * time_vals)  # X-axis
        mag[:, 1] = 5 * np.sin(2 * np.pi * 0.2 * time_vals)       # Y-axis
        mag[:, 2] = 10 + 2 * np.random.randn(n_samples)           # Z-axis
    
    return time_vals, accel, gyro, mag

def main():
    # Configuration
    imu_type = '6dof'  # Change to '9dof' for magnetometer support
    duration = 5       # seconds
    freq = 100         # Hz
    
    # Initialize components
    processor = IMUProcessor()
    visualizer = IMUVisualizer()
    data_io = IMUDataIO()
    
    # Generate or load IMU data
    time_vals, accel, gyro, mag = generate_simulated_data(duration, freq, imu_type)
    
    # Process data in batch
    data = {
        'time': time_vals,
        'accel': accel,
        'gyro': gyro
    }
    if mag is not None:
        data['mag'] = mag
        
    processed = processor.process_imu_data(data)
    orientations = processed['orientation']
    positions = processed['position']
    
    # Visualize results
    raw_data = {
        'time': time_vals,
        'accel': accel,
        'gyro': gyro,
        'mag': mag
    }
    visualizer.plot_raw_data(raw_data)
    
    orientation_data = {
        'time': time_vals,
        'orientation': orientations
    }
    visualizer.plot_orientation(orientation_data)
    
    trajectory_data = {
        'position': positions
    }
    visualizer.plot_trajectory(trajectory_data)
    visualizer.show_all()
    
    # Save data
    processed_data = {
        'time': time_vals,
        'accel': accel,
        'gyro': gyro,
        'mag': mag,
        'orientation': orientations,
        'position': positions,
        'velocity': np.zeros_like(positions)  # Dummy velocity data
    }
    data_io.save_results(processed_data, ".", "imu_data")
    
    print("Processing complete. Data saved to text files.")

if __name__ == "__main__":
    print("Start")
    main()
