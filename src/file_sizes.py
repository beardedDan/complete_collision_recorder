import os
import csv
import fnmatch

def load_gitignore_patterns(directory):
    gitignore_path = os.path.join(directory, '.gitignore')
    patterns = []
    
    # Check if the .gitignore file exists
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as gitignore_file:
            for line in gitignore_file:
                # Strip whitespace and ignore comments and empty lines
                line = line.strip()
                if line and not line.startswith('#'):
                    # Make sure directory patterns like .venv/ work correctly
                    if line.endswith('/'):
                        line += '*'  # Ensure it matches everything in the directory
                    patterns.append(line)
    
    return patterns

def is_file_ignored(file_path, patterns, root_dir):
    # Convert to relative path
    relative_path = os.path.relpath(file_path, root_dir)
    
    # Check for both exact matches and subdirectory matches
    for pattern in patterns:
        if fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(relative_path, f'*/{pattern}'):
            return True
    return False

def get_files_size_to_csv(directory, output_csv):
    # Load .gitignore patterns
    gitignore_patterns = load_gitignore_patterns(directory)

    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['File Path', 'File Size (MB)'])
        
        # Walk through directory and subdirectories
        for root, dirs, files in os.walk(directory):
            for file_name in files:
                # Full path of the file
                file_path = os.path.join(root, file_name)
                
                # Check if the file matches any .gitignore patterns
                if is_file_ignored(file_path, gitignore_patterns, directory):
                    continue  # Skip files in .gitignore
                
                try:
                    # Get the size of the file
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                    # Write the file path and size to the CSV file
                    writer.writerow([file_path, f"{file_size:.2f}"])
                except OSError as e:
                    print(f"Error accessing file {file_path}: {e}")

# Specify the directory you want to check and the output CSV file path
directory_path = "../"
output_csv = "file_sizes_filtered.csv"

# Call the function to save the file sizes to the CSV, excluding .gitignore files
get_files_size_to_csv(directory_path, output_csv)

print(f"Filtered file sizes have been saved to {output_csv}")
