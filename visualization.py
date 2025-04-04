import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
from typing import Dict, Optional
import os
from pathlib import Path

class IMUVisualizer:
    def __init__(self):
        """Initialize IMU data visualizer"""
        plt.style.use('seaborn-v0_8')
        self.figures = []
        
    def plot_raw_data(self, imu_data: Dict, save_path: Optional[str] = None):
        """
        Plot raw IMU data
        
        Parameters:
        imu_data: Dictionary containing raw IMU data
        save_path: Path to save the plot (optional)
        """
        time = imu_data['time']
        accel = imu_data['accel']
        gyro = imu_data['gyro']
        mag = imu_data.get('mag')
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle('Raw IMU Data', fontsize=16)
        
        # Plot accelerometer data
        axes[0].plot(time, accel[:, 0], label='X')
        axes[0].plot(time, accel[:, 1], label='Y')
        axes[0].plot(time, accel[:, 2], label='Z')
        axes[0].set_title('Accelerometer')
        axes[0].set_ylabel('Acceleration (m/s²)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot gyroscope data
        axes[1].plot(time, gyro[:, 0], label='X')
        axes[1].plot(time, gyro[:, 1], label='Y')
        axes[1].plot(time, gyro[:, 2], label='Z')
        axes[1].set_title('Gyroscope')
        axes[1].set_ylabel('Angular Rate (rad/s)')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot magnetometer data if available
        if mag is not None:
            axes[2].plot(time, mag[:, 0], label='X')
            axes[2].plot(time, mag[:, 1], label='Y')
            axes[2].plot(time, mag[:, 2], label='Z')
            axes[2].set_title('Magnetometer')
            axes[2].set_ylabel('Magnetic Field (μT)')
            axes[2].set_xlabel('Time (s)')
            axes[2].legend()
            axes[2].grid(True)
        else:
            fig.delaxes(axes[2])
            
        plt.tight_layout()
        self.figures.append(fig)
        
        if save_path:
            self._save_figure(fig, save_path, 'raw_data')
            
        return fig
        
    def plot_orientation(self, 
                        processed_data: Dict, 
                        save_path: Optional[str] = None):
        """
        Plot computed orientation angles
        
        Parameters:
        processed_data: Dictionary containing processed data
        save_path: Path to save the plot (optional)
        """
        time = processed_data['time']
        orientation = np.degrees(processed_data['orientation'])  # Convert to degrees
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle('Orientation Angles', fontsize=16)
        
        ax.plot(time, orientation[:, 0], label='Roll')
        ax.plot(time, orientation[:, 1], label='Pitch')
        ax.plot(time, orientation[:, 2], label='Yaw')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (deg)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        self.figures.append(fig)
        
        if save_path:
            self._save_figure(fig, save_path, 'orientation')
            
        return fig
        
    def plot_trajectory(self, 
                       processed_data: Dict,
                       save_path: Optional[str] = None,
                       interactive: bool = True):
        """
        Plot 3D trajectory
        
        Parameters:
        processed_data: Dictionary containing processed data
        save_path: Path to save the plot (optional)
        interactive: Whether to enable interactive controls
        """
        position = processed_data['position']
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle('3D Trajectory', fontsize=16)
        
        # Plot trajectory
        line, = ax.plot(position[:, 0], position[:, 1], position[:, 2], 
                       'b-', linewidth=2, label='Trajectory')
        
        # Plot start and end points
        ax.scatter(position[0, 0], position[0, 1], position[0, 2], 
                  c='g', s=100, label='Start')
        ax.scatter(position[-1, 0], position[-1, 1], position[-1, 2], 
                  c='r', s=100, label='End')
                  
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        
        # Equal aspect ratio
        max_range = np.array([
            position[:, 0].max()-position[:, 0].min(), 
            position[:, 1].max()-position[:, 1].min(),
            position[:, 2].max()-position[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (position[:, 0].max()+position[:, 0].min()) * 0.5
        mid_y = (position[:, 1].max()+position[:, 1].min()) * 0.5
        mid_z = (position[:, 2].max()+position[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if interactive:
            # Add slider for time window
            ax_time = plt.axes([0.25, 0.1, 0.65, 0.03])
            time_slider = Slider(
                ax=ax_time,
                label='Time Window',
                valmin=0,
                valmax=len(position)-1,
                valinit=len(position)-1,
                valstep=1
            )
            
            def update(val):
                idx = int(time_slider.val)
                line.set_data(position[:idx, 0], position[:idx, 1])
                line.set_3d_properties(position[:idx, 2])
                fig.canvas.draw_idle()
                
            time_slider.on_changed(update)
            
            # Add reset button
            reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
            reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')
            
            def reset(event):
                time_slider.reset()
                
            reset_button.on_clicked(reset)
            
        plt.tight_layout()
        self.figures.append(fig)
        
        if save_path:
            self._save_figure(fig, save_path, 'trajectory')
            
        return fig
        
    def show_all(self):
        """Show all created figures"""
        plt.show()
        
    def close_all(self):
        """Close all figures"""
        for fig in self.figures:
            plt.close(fig)
        self.figures = []
        
    def _save_figure(self, fig, save_dir: str, name: str):
        """
        Save figure to file
        
        Parameters:
        save_dir: Directory to save the figure
        name: Name of the figure
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = str(Path(save_dir) / f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {path}")
