
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to data
DATA_DIR = r"f:\数模\美赛\Code\data\mobile data"

def load_first_valid_discharge_segment(min_duration_sec=1800, min_soc_drop=20):
    """
    Finds the first file with a valid discharge segment.
    """
    # Search for all dynamic processed files
    search_pattern = os.path.join(DATA_DIR, "**", "*_dynamic_processed.csv")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} files. Scanning for discharge segments...")
    
    for file_path in files:
        try:
            df = pd.read_csv(file_path, low_memory=False)
            
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter for discharging status (assuming 3 is discharging based on base.py)
            # Also check if current is negative or positive. Usually discharging current is negative, 
            # but sometimes reported as positive. We need to check.
            # In base.py: battery_charging_status == 3
            
            discharge_df = df[df['battery_charging_status'] == 3].copy()
            
            if len(discharge_df) < 100:
                continue
                
            # Find continuous segments
            discharge_df['time_diff'] = discharge_df['timestamp'].diff().dt.total_seconds()
            # If time diff > 60s, it's a new segment
            discharge_df['segment_id'] = (discharge_df['time_diff'] > 60).cumsum()
            
            for seg_id, segment in discharge_df.groupby('segment_id'):
                duration = (segment['timestamp'].iloc[-1] - segment['timestamp'].iloc[0]).total_seconds()
                soc_start = segment['battery_level'].iloc[0]
                soc_end = segment['battery_level'].iloc[-1]
                soc_drop = soc_start - soc_end
                
                if duration >= min_duration_sec and soc_drop >= min_soc_drop:
                    print(f"Found valid segment in {os.path.basename(file_path)}")
                    print(f"  Duration: {duration/60:.1f} min")
                    print(f"  SOC: {soc_start}% -> {soc_end}% (Drop: {soc_drop}%)")
                    return segment, file_path
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
    print("No valid discharge segment found.")
    return None, None

def estimate_q0(segment):
    """
    Estimates Q0 using Ampere-hour integration method.
    Q_real = Integral(I dt) / (SOC_start - SOC_end)
    """
    # Sort by time
    segment = segment.sort_values('timestamp')
    
    # Calculate time differences in hours
    # We use trapezoidal rule for integration
    time_seconds = (segment['timestamp'] - segment['timestamp'].iloc[0]).dt.total_seconds()
    time_hours = time_seconds / 3600.0
    
    # Current (Amps)
    # Note: We need to ensure current is positive for discharge calculation if we treat it as "released charge"
    # Or if current is negative during discharge, we take absolute value.
    # Let's check the mean current.
    mean_current = segment['battery_current'].mean()
    print(f"  Mean current in segment: {mean_current:.4f} A")
    
    # Usually in these datasets, discharge current might be positive or negative.
    # If status is discharging, we assume the current represents the load.
    # We'll take the absolute value to be safe for "released charge".
    current_abs = segment['battery_current'].abs()
    
    # Integrate Current over Time (Ah)
    # using numpy trapz
    discharged_capacity_Ah = np.trapz(current_abs, x=time_hours)
    
    # SOC change (fraction 0-1)
    soc_start = segment['battery_level'].iloc[0]
    soc_end = segment['battery_level'].iloc[-1]
    delta_soc = (soc_start - soc_end) / 100.0
    
    if delta_soc == 0:
        return 0
        
    # Q_real = Discharged Ah / Delta SOC
    q_real = discharged_capacity_Ah / delta_soc
    
    return q_real, discharged_capacity_Ah, delta_soc

def main():
    segment, file_path = load_first_valid_discharge_segment()
    
    if segment is not None:
        q_real, discharged_Ah, delta_soc = estimate_q0(segment)
        
        print("\n" + "="*40)
        print(f"Estimation Result for {os.path.basename(file_path)}")
        print("="*40)
        print(f"Discharged Capacity: {discharged_Ah:.4f} Ah")
        print(f"SOC Change: {delta_soc*100:.1f}%")
        print(f"Estimated Q0 (Total Capacity): {q_real:.4f} Ah")
        print(f"Estimated Q0 (Total Capacity): {q_real*1000:.1f} mAh")
        print("="*40)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(segment['timestamp'], segment['battery_level'], 'b-')
        plt.ylabel('SOC (%)')
        plt.title('SOC vs Time')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(segment['timestamp'], segment['battery_current'], 'r-')
        plt.ylabel('Current (A)')
        plt.xlabel('Time')
        plt.title('Current vs Time')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('q0_estimation.png')
        print("Saved plot to q0_estimation.png")

if __name__ == "__main__":
    main()
