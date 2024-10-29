import os
import toml
import subprocess
import sys

def install_packages(pyproject_path=None):
    pip_path = os.path.join(os.path.dirname(sys.executable), 'pip')

    if pyproject_path is None:
        # Set pyproject.toml to the project root, assuming this script is in the 'bin' directory
        pyproject_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pyproject.toml')


    # Check if pyproject.toml exists
    if not os.path.exists(pyproject_path):
        print(f"{pyproject_path} not found.")
        return
    
    # Load the pyproject.toml file
    with open(pyproject_path, 'r') as file:
        pyproject_data = toml.load(file)
    
    # Extract dependencies from [complete_collision_recorder.dependencies]
    dependencies = pyproject_data.get('complete_collision_recorder', {}).get('dependencies', {})

    if not dependencies:
        print("No dependencies found in pyproject.toml.")
        return

    # Install each dependency from [complete_collision_recorder.dependencies]
    for package, version in dependencies.items():
        if package == 'python':  # Skip the Python version specification
            continue
        package_to_install = f"{package}{version}" if isinstance(version, str) else package
        print(f"Installing {package_to_install}...")
        subprocess.check_call([pip_path, "install", package_to_install])
        print(f"Installed {package_to_install} successfully.")

    print("All dependencies from pyproject.toml have been installed.") 


    # Extract optional dependencies from [complete_collision_recorder.optional-dependencies]
    optional_dependencies = pyproject_data.get('complete_collision_recorder', {}).get('optional-dependencies', {})

    if not optional_dependencies:
        print("No optional dependencies found in pyproject.toml.")
        return

    # Install each optional dependency (assumed to be a list)
    for group, packages in optional_dependencies.items():
        print(f"Installing optional dependencies from the '{group}' group...")
        for package in packages:
            print(f"Installing {package}...")
            subprocess.check_call([pip_path, "install", package])
            print(f"Installed {package} successfully.")

    print("All optional dependencies from pyproject.toml have been installed.")

if __name__ == "__main__":
    install_packages()
