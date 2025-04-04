import numpy as np
from typing import Dict, Tuple, Optional
import re
from pathlib import Path
import json

class IMUDataIO:
    def __init__(self):
        """Initialize IMU data input/output handler"""
        self.supported_formats = {
            'csv': self._load_csv,
            'txt': self._load_txt,
            'json': self._load_json
        }
        
    def load_imu_data(self, file_path: str) -> Dict:
        """
        Load IMU data from file
        
        Parameters:
        file_path: path to IMU data file
        
        Returns:
        Dictionary containing loaded IMU data
        """
        file_ext = Path(file_path).suffix[1:].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        return self.supported_formats[file_ext](file_path)
        
    def save_results(self, 
                    processed_data: Dict, 
                    output_dir: str,
                    prefix: str = 'imu_result') -> Tuple[str, str]:
        """
        Save processed data to output directory
        
        Parameters:
        processed_data: dictionary containing processed data
        output_dir: directory to save results
        prefix: prefix for output filenames
        
        Returns:
        Tuple of (orientation_file_path, trajectory_file_path)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save orientation data
        orient_path = str(Path(output_dir) / f"{prefix}_orientation.txt")
        self._save_orientation(orient_path, processed_data)
        
        # Save trajectory data
        traj_path = str(Path(output_dir) / f"{prefix}_trajectory.txt")
        self._save_trajectory(traj_path, processed_data)
        
        return orient_path, traj_path
        
    def _load_csv(self, file_path: str) -> Dict:
        """Load IMU data from CSV file"""
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        
        # Try to detect column order
        with open(file_path, 'r') as f:
            header = f.readline().strip().lower()
            
        # Common column patterns
        patterns = {
            'time': r'time|timestamp',
            'accel': r'accel|accelerometer|ax|ay|az',
            'gyro': r'gyro|gyroscope|gx|gy|gz',
            'mag': r'mag|magnetometer|mx|my|mz'
        }
        
        columns = {}
        for col_type, pattern in patterns.items():
            matches = re.finditer(pattern, header)
            cols = [m.start() for m in matches]
            if cols:
                columns[col_type] = cols[0] if len(cols) == 1 else cols
                
        # Default column mapping if not detected
        if not columns:
            columns = {
                'time': 0,
                'accel': [1, 2, 3],
                'gyro': [4, 5, 6],
                'mag': [7, 8, 9] if data.shape[1] > 7 else None
            }
            
        # Extract data
        imu_data = {
            'time': data[:, columns['time']]
        }
        
        if isinstance(columns['accel'], list):
            imu_data['accel'] = data[:, columns['accel']]
        else:
            imu_data['accel'] = data[:, columns['accel']:columns['accel']+3]
            
        if isinstance(columns['gyro'], list):
            imu_data['gyro'] = data[:, columns['gyro']]
        else:
            imu_data['gyro'] = data[:, columns['gyro']:columns['gyro']+3]
            
        if columns.get('mag'):
            if isinstance(columns['mag'], list):
                imu_data['mag'] = data[:, columns['mag']]
            else:
                imu_data['mag'] = data[:, columns['mag']:columns['mag']+3]
                
        return imu_data
        
    def _load_txt(self, file_path: str) -> Dict:
        """Load IMU data from space/tab-delimited text file"""
        data = np.loadtxt(file_path)
        
        # Assume standard column order for text files
        imu_data = {
            'time': data[:, 0],
            'accel': data[:, 1:4],
            'gyro': data[:, 4:7]
        }
        
        if data.shape[1] > 7:
            imu_data['mag'] = data[:, 7:10]
            
        return imu_data
        
    def _load_json(self, file_path: str) -> Dict:
        """Load IMU data from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Convert lists to numpy arrays
        imu_data = {}
        for key, value in data.items():
            if isinstance(value, list):
                imu_data[key] = np.array(value)
            else:
                imu_data[key] = value
                
        return imu_data
        
    def _save_orientation(self, file_path: str, processed_data: Dict):
        """Save orientation data to file"""
        header = "Time(s)\tRoll(deg)\tPitch(deg)\tYaw(deg)"
        data = np.column_stack([
            processed_data['time'],
            np.degrees(processed_data['orientation'])
        ])
        np.savetxt(file_path, data, 
                  header=header, delimiter='\t', fmt='%.6f', comments='')
        
    def _save_trajectory(self, file_path: str, processed_data: Dict):
        """Save trajectory data to file"""
        header = "Time(s)\tX(m)\tY(m)\tZ(m)\tVx(m/s)\tVy(m/s)\tVz(m/s)"
        data = np.column_stack([
            processed_data['time'],
            processed_data['position'],
            processed_data['velocity']
        ])
        np.savetxt(file_path, data, 
                  header=header, delimiter='\t', fmt='%.6f', comments='')
        
    def convert_to_standard_format(self, 
                                 imu_data: Dict,
                                 output_format: str = 'csv') -> str:
        """
        Convert IMU data to standard format
        
        Parameters:
        imu_data: dictionary containing IMU data
        output_format: output format ('csv', 'txt', 'json')
        
        Returns:
        String containing converted data
        """
        if output_format == 'csv':
            header = "time,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z"
            if 'mag' in imu_data:
                header += ",mag_x,mag_y,mag_z"
                
            data = [imu_data['time']]
            data.extend([
                imu_data['accel'][:, 0], imu_data['accel'][:, 1], imu_data['accel'][:, 2],
                imu_data['gyro'][:, 0], imu_data['gyro'][:, 1], imu_data['gyro'][:, 2]
            ])
            
            if 'mag' in imu_data:
                data.extend([
                    imu_data['mag'][:, 0], imu_data['mag'][:, 1], imu_data['mag'][:, 2]
                ])
                
            return header + '\n' + '\n'.join(
                ','.join(f"{x:.6f}" for x in row) for row in zip(*data))
                
        elif output_format == 'txt':
            header = "# time accel_x accel_y accel_z gyro_x gyro_y gyro_z"
            if 'mag' in imu_data:
                header += " mag_x mag_y mag_z"
                
            data = [imu_data['time']]
            data.extend([
                imu_data['accel'][:, 0], imu_data['accel'][:, 1], imu_data['accel'][:, 2],
                imu_data['gyro'][:, 0], imu_data['gyro'][:, 1], imu_data['gyro'][:, 2]
            ])
            
            if 'mag' in imu_data:
                data.extend([
                    imu_data['mag'][:, 0], imu_data['mag'][:, 1], imu_data['mag'][:, 2]
                ])
                
            return header + '\n' + '\n'.join(
                ' '.join(f"{x:.6f}" for x in row) for row in zip(*data))
                
        elif output_format == 'json':
            data = {
                'time': imu_data['time'].tolist(),
                'accel': imu_data['accel'].tolist(),
                'gyro': imu_data['gyro'].tolist()
            }
            
            if 'mag' in imu_data:
                data['mag'] = imu_data['mag'].tolist()
                
            return json.dumps(data, indent=2)
            
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
