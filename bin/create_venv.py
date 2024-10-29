import os
import subprocess
import sys
import venv

def create_virtual_env(root_folder, env_name=".venv"):
    # Define the full path for the virtual environment
    venv_path = os.path.join(root_folder, env_name)
    
    # Check if the virtual environment already exists
    if os.path.exists(venv_path):
        print(f"Virtual environment '{env_name}' already exists in {root_folder}.")
        return
    
    # Create the virtual environment
    print(f"Creating virtual environment '{env_name}' in {root_folder}...")
    venv.create(venv_path, with_pip=True)
    print(f"Virtual environment created at {venv_path}.")
    
    # Install toml inside the virtual environment
    install_toml(venv_path)

    # Run the second script to install packages from pyproject.toml
    run_install_packages_in_venv(venv_path)


def install_toml(venv_path):
    # Path to pip in the virtual environment
    pip_path = os.path.join(venv_path, 'bin', 'pip') if sys.platform != "win32" else os.path.join(venv_path, 'Scripts', 'pip')
    
    # Install the toml package
    print("Installing toml package in the virtual environment...")
    subprocess.check_call([pip_path, "install", "toml"])
    print("toml package installed.")


def run_install_packages_in_venv(venv_path):
    # Path to the Python interpreter inside the virtual environment
    python_path = os.path.join(venv_path, 'bin', 'python') if sys.platform != "win32" else os.path.join(venv_path, 'Scripts', 'python')

    # Run the install_dependencies.py script using the virtual environment's Python interpreter
    print(f"Running package installation inside the virtual environment at {venv_path}...")
    subprocess.check_call([python_path, os.path.join(os.path.dirname(__file__), "install_dependencies.py")])



if __name__ == "__main__":
    root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    create_virtual_env(root_folder)
