import os, sys
import tarfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# file_path = "dataset.tar.gz"
# extract_path = "./core"
# with tarfile.open(file_path, "r:gz") as tar:
#     tar.extractall(path=extract_path)
#     print(f"Extracted all contents to {extract_path}.")
    
import psutil

def print_cpu_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"RSS Memory: {mem_info.rss / 1e6} MB")
