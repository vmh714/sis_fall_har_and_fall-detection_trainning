import os
import numpy as np
import pandas as pd
from scipy import signal
from dotenv import load_dotenv

def process_sisfall_file(input_path, output_path=None):
    """
    Read IMU data from SisFall dataset file, downsample from 200Hz to 100Hz,
    and convert bit values to physical units (g and degree/s) based on
    the information provided in Readme.txt.
    """
    try:
        # Read the file line by line and parse manually to handle trailing semicolons
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        data = []
        for line in lines:
            line = line.strip().strip(';')
            if not line: continue
            row = [float(x) for x in line.split(',')]
            data.append(row)
            
        df = pd.DataFrame(data, columns=[
            'ADXL_X', 'ADXL_Y', 'ADXL_Z', 
            'ITG_X', 'ITG_Y', 'ITG_Z', 
            'MMA_X', 'MMA_Y', 'MMA_Z'
        ])
    except Exception as e:
        print(f"Error reading file {input_path}: {e}")
        return None

    # 1. Downsample from 200Hz to 100Hz safely
    # Using scipy.signal.decimate which applies an order 8 Chebyshev type I 
    # low-pass filter (anti-aliasing) before downsampling by a factor of 2.
    downsampled_data = signal.decimate(df.values, q=2, axis=0)
    df_100hz = pd.DataFrame(downsampled_data, columns=df.columns)

    # 2. Parse units
    # ADXL345 (Accelerometer): Resolution 13 bits, Range +-16g
    # Formula: [(2*16)/(2^13)] * AD = (32 / 8192) * AD = 0.00390625 * AD
    adxl_factor = (2 * 16) / (2 ** 13)
    df_100hz[['ADXL_X', 'ADXL_Y', 'ADXL_Z']] = df_100hz[['ADXL_X', 'ADXL_Y', 'ADXL_Z']] * adxl_factor

    # ITG3200 (Gyroscope): Resolution 16 bits, Range +-2000 degree/s
    # Formula: [(2*2000)/(2^16)] * RD = (4000 / 65536) * RD = 0.06103515625 * RD
    itg_factor = (2 * 2000) / (2 ** 16)
    df_100hz[['ITG_X', 'ITG_Y', 'ITG_Z']] = df_100hz[['ITG_X', 'ITG_Y', 'ITG_Z']] * itg_factor

    # MMA8451Q (Accelerometer): Resolution 14 bits, Range +-8g
    # Formula: [(2*8)/(2^14)] * AD = (16 / 16384) * AD = 0.0009765625 * AD
    mma_factor = (2 * 8) / (2 ** 14)
    df_100hz[['MMA_X', 'MMA_Y', 'MMA_Z']] = df_100hz[['MMA_X', 'MMA_Y', 'MMA_Z']] * mma_factor

    if output_path:
        df_100hz.to_csv(output_path, index=False)
        print(f"Saved processed data to: {output_path}")

    return df_100hz

import glob

def process_all_dataset(input_dir, output_dir):
    """
    Process all .txt files in input_dir and its subdirectories, 
    and save the processed .csv files to output_dir while maintaining
    the original directory structure.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Find all .txt files recursively
    search_pattern = os.path.join(input_dir, '**', '*.txt')
    txt_files = glob.glob(search_pattern, recursive=True)
    
    processed_count = 0
    for txt_file in txt_files:
        # Skip Readme.txt
        if os.path.basename(txt_file).lower() == 'readme.txt':
            continue
            
        # Get relative path to maintain folder structure (e.g., SA01\D01_SA01_R01.txt)
        rel_path = os.path.relpath(txt_file, input_dir)
        
        # Create corresponding output directory
        out_file_path = os.path.join(output_dir, rel_path)
        out_file_path = os.path.splitext(out_file_path)[0] + '.csv'
        out_file_dir = os.path.dirname(out_file_path)
        
        if not os.path.exists(out_file_dir):
            os.makedirs(out_file_dir)
            
        # Process and save file
        print(f"Processing: {rel_path} -> {out_file_path}")
        process_sisfall_file(txt_file, out_file_path)
        processed_count += 1
        
    print(f"Finished processing {processed_count} files.")
if __name__ == '__main__':
    # Load biến môi trường từ file .env
    load_dotenv()
    
    # Lấy đường dẫn gốc của project từ .env, nếu không có thì mặc định lấy thư mục chứa file script này
    env_root = os.environ.get('PROJECT_ROOT')
    if env_root and os.path.exists(env_root):
        base_dir = env_root
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
    input_dataset_dir = os.path.join(base_dir, "SisFall_dataset")
    output_dataset_dir = os.path.join(base_dir, "SisFall_dataset_Processed")
    
    if os.path.exists(input_dataset_dir):
        print(f"Starting to process all files from {input_dataset_dir} to {output_dataset_dir}")
        process_all_dataset(input_dataset_dir, output_dataset_dir)
    else:
        print(f"Input directory not found: {input_dataset_dir}")
