#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
System Performance Monitor for Facial Authentication
This utility monitors system resources (CPU, memory, temperature) during facial authentication.
Optimized for Jetson Orin Nano X hardware monitoring.
"""

import os
import time
import threading
import psutil
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime

class SystemMonitor:
    """Monitors system performance metrics during facial authentication."""
    
    def __init__(self, output_dir=None, sampling_interval=1.0):
        """
        Initialize the system monitor.
        
        Args:
            output_dir: Directory to save monitoring data
            sampling_interval: Time between samples in seconds
        """
        self.sampling_interval = sampling_interval
        
        # Create output directory if specified
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitoring")
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data collection arrays
        self.timestamps = []
        self.cpu_usages = []
        self.memory_usages = []
        self.temperatures = []
        
        # Control flag for monitoring thread
        self.is_monitoring = False
        self.monitor_thread = None
        
        # For Jetson Orin monitoring
        self.is_jetson = self._check_if_jetson()
    
    def _check_if_jetson(self):
        """Check if running on a Jetson device."""
        try:
            # Check for Jetson-specific paths
            return os.path.exists('/sys/devices/platform/host1x')
        except:
            return False
    
    def _get_jetson_temperatures(self):
        """Get temperature readings from Jetson thermal zones."""
        temps = {}
        
        try:
            if self.is_jetson:
                # Common thermal zone paths on Jetson platforms
                thermal_zones = [
                    ('/sys/devices/virtual/thermal/thermal_zone0/temp', 'CPU'),
                    ('/sys/devices/virtual/thermal/thermal_zone1/temp', 'GPU'),
                    ('/sys/devices/virtual/thermal/thermal_zone2/temp', 'PLL')
                ]
                
                for path, name in thermal_zones:
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            # Convert to Celsius (value is in milliCelsius)
                            temp = float(f.read().strip()) / 1000.0
                            temps[name] = temp
            
            # Add CPU temperature on non-Jetson platforms where available
            if not temps and hasattr(psutil, 'sensors_temperatures'):
                sensor_temps = psutil.sensors_temperatures()
                if sensor_temps:
                    for chip, values in sensor_temps.items():
                        for entry in values:
                            temps[f"{chip}_{entry.label or 'CPU'}"] = entry.current
        except Exception as e:
            print(f"Error reading temperatures: {e}")
        
        return temps
    
    def _get_jetson_power_usage(self):
        """Get power usage information from Jetson."""
        power_info = {}
        
        try:
            if self.is_jetson:
                # Path for Jetson power consumption (may vary by model)
                power_paths = [
                    '/sys/bus/i2c/drivers/ina3221x/*/iio_device/in_power*_input',
                    '/sys/devices/platform/host1x/15040000.nvjpg/power/runtime_active_time'
                ]
                
                import glob
                for pattern in power_paths:
                    files = glob.glob(pattern)
                    for i, path in enumerate(files):
                        try:
                            with open(path, 'r') as f:
                                # Power is usually in microwatts or milliwatts
                                power = float(f.read().strip()) / 1000.0  # Convert to watts
                                power_info[f"Power_{i}"] = power
                        except:
                            pass
        except Exception as e:
            print(f"Error reading power information: {e}")
        
        return power_info
    
    def _monitor_system(self):
        """Monitor system metrics at regular intervals."""
        while self.is_monitoring:
            timestamp = time.time()
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get temperatures where available
            temps = self._get_jetson_temperatures()
            
            # Store the data
            self.timestamps.append(timestamp)
            self.cpu_usages.append(cpu_percent)
            self.memory_usages.append(memory_percent)
            self.temperatures.append(temps)
            
            # Wait for next sample
            time.sleep(self.sampling_interval)
    
    def start_monitoring(self):
        """Start the system monitoring thread."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_system)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print(f"System monitoring started. Sampling interval: {self.sampling_interval}s")
    
    def stop_monitoring(self):
        """Stop the system monitoring thread and save data."""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2.0)
            
            # Save the monitoring data
            self._save_monitoring_data()
            print("System monitoring stopped. Data saved to:", self.output_dir)
    
    def _save_monitoring_data(self):
        """Save the monitoring data to CSV and generate plots."""
        if not self.timestamps:
            print("No monitoring data to save.")
            return
        
        # Create unique filename based on timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV data
        csv_path = os.path.join(self.output_dir, f"system_metrics_{timestamp_str}.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['Timestamp', 'CPU_Usage', 'Memory_Usage']
            
            # Add temperature sensor names to header
            if self.temperatures and self.temperatures[0]:
                for sensor in self.temperatures[0].keys():
                    header.append(f'Temp_{sensor}')
            
            writer.writerow(header)
            
            # Write data rows
            for i, timestamp in enumerate(self.timestamps):
                row = [timestamp, self.cpu_usages[i], self.memory_usages[i]]
                
                # Add temperature values
                if i < len(self.temperatures) and self.temperatures[i]:
                    for sensor in header[3:]:  # Skip Timestamp, CPU, Memory
                        sensor_name = sensor[5:]  # Remove 'Temp_' prefix
                        row.append(self.temperatures[i].get(sensor_name, 'N/A'))
                
                writer.writerow(row)
        
        # Generate plots
        self._generate_plots(timestamp_str)
    
    def _generate_plots(self, timestamp_str):
        """Generate performance monitoring plots."""
        if not self.timestamps:
            return
        
        # Convert timestamps to relative seconds from start
        rel_times = [t - self.timestamps[0] for t in self.timestamps]
        
        plt.figure(figsize=(12, 8))
        
        # Plot CPU usage
        plt.subplot(2, 1, 1)
        plt.plot(rel_times, self.cpu_usages, 'b-', label='CPU Usage (%)')
        plt.plot(rel_times, self.memory_usages, 'g-', label='Memory Usage (%)')
        plt.title('System Resource Usage During Facial Authentication')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Usage (%)')
        plt.grid(True)
        plt.legend()
        
        # Plot temperature if available
        if self.temperatures and any(self.temperatures):
            plt.subplot(2, 1, 2)
            
            # Extract temperature data for each sensor
            sensor_data = {}
            for i, temp_dict in enumerate(self.temperatures):
                for sensor, value in temp_dict.items():
                    if sensor not in sensor_data:
                        sensor_data[sensor] = []
                    sensor_data[sensor].append(value)
            
            # Plot each temperature sensor
            for sensor, values in sensor_data.items():
                # Pad with None if lengths don't match
                while len(values) < len(rel_times):
                    values.append(None)
                plt.plot(rel_times[:len(values)], values, label=f'{sensor} (°C)')
            
            plt.title('System Temperatures')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Temperature (°C)')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, f"performance_plot_{timestamp_str}.png")
        plt.savefig(plot_path)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="System Performance Monitor")
    parser.add_argument('--output', type=str, default=None,
                       help='Directory to save monitoring data')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Sampling interval in seconds')
    return parser.parse_args()


def main():
    args = parse_args()
    
    monitor = SystemMonitor(output_dir=args.output, sampling_interval=args.interval)
    
    try:
        monitor.start_monitoring()
        
        # Keep running until user interrupts
        print("System monitoring active. Press Ctrl+C to stop...")
        while True:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    finally:
        monitor.stop_monitoring()
        print("Monitoring complete.")


if __name__ == "__main__":
    main()
