import numpy as np

def load_and_print_npy(file_path):
    data = np.load(file_path, allow_pickle=True)
    
    # 
    print(f"Data type: {type(data)}")
    print(f"Shape of data: {np.shape(data)}")
    
    # 
    if data.shape == ():
        print(f"Data content: {data}")
    #
    elif isinstance(data, dict):
        for key, value in data.items():
            print(f"\nKey: {key}")
            print(f"Type of value: {type(value)}")
            print(f"Shape of value: {np.shape(value)}")
            if isinstance(value, np.ndarray):
                print(f"Sample content (first 3 elements): {value[:3]}")
            else:
                print(f"Content: {value}")
    # 
    elif isinstance(data, np.ndarray):
        print(f"Data is an ndarray with shape: {np.shape(data)}")
        print(f"First few elements: {data[:3]}")
def main():
    # 
    file_path = '/home/rose/DexGraspBench/output/debug_shadow/succgrasp/core_bowl_4967063fc3673caa47fe6b02985fbd0/scale012_pose000/0.npy' \
    ''  # 替换为你文件的实际路径
    load_and_print_npy(file_path)

if __name__ == "__main__":
    main()
