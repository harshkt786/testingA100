import sys
import subprocess

def run_command(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def check_package(package_name):
    return run_command(f"{sys.executable} -m pip show {package_name}")

def check_cuda():
    return run_command("nvidia-smi")

def check_pytorch_cuda():
    command = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU(s) available: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
"""
    return run_command(f"{sys.executable} -c '{command}'")

def main():
    print("Checking A100 GPU Environment:\n")

    print("1. CUDA Availability:")
    print(check_cuda())
    print("\n2. Python version:")
    print(sys.version)
    
    packages_to_check = ['torch', 'tensorflow', 'numpy', 'scipy', 'pandas', 'sklearn']
    
    print("\n3. Installed Packages:")
    for package in packages_to_check:
        result = check_package(package)
        if "Version" in result:
            version = result.split('\n')[1].split(': ')[1]
            print(f"{package}: {version}")
        else:
            print(f"{package}: Not installed")
    
    print("\n4. PyTorch CUDA Support:")
    print(check_pytorch_cuda())

if __name__ == "__main__":
    main()
