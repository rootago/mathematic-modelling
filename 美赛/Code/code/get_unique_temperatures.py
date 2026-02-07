import os
import pandas as pd
import glob

def get_unique_temperatures(root_dir):
    unique_temps = set()
    
    # Search for all dynamic processed csv files
    # Pattern matching: recursively find files ending with _dynamic_processed.csv
    search_pattern = os.path.join(root_dir, "**", "*_dynamic_processed.csv")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} files to process.")
    
    for file_path in files:
        try:
            # Read only the battery_temperature column to save memory
            df = pd.read_csv(file_path, usecols=['battery_temperature'])
            
            # Add unique values from this file to the set
            unique_temps.update(df['battery_temperature'].unique())
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            # Try checking if column name is different or file is empty
            try:
                df = pd.read_csv(file_path, nrows=1)
                if 'battery_temperature' not in df.columns:
                    print(f"Column 'battery_temperature' not found in {file_path}. Columns: {df.columns.tolist()}")
            except:
                pass

    # Convert to sorted list
    sorted_temps = sorted(list(unique_temps))
    
    print("\nUnique Battery Temperatures found:")
    for temp in sorted_temps:
        print(temp)
        
    return sorted_temps

if __name__ == "__main__":
    # Path to the mobile data directory
    # Using raw string for Windows path
    data_dir = r"f:\数模\美赛\Code\data\mobile data"
    
    if os.path.exists(data_dir):
        get_unique_temperatures(data_dir)
    else:
        print(f"Directory not found: {data_dir}")
