import os

def find_large_files(directory, size_limit_mb):
    size_limit = size_limit_mb * 1024 * 1024  # Convert MB to bytes
    for root, dirs, files in os.walk(directory):
        # Skip `.venv` directories
        dirs[:] = [d for d in dirs if d != '.venv']
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if os.path.getsize(file_path) > size_limit:
                    size_in_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"{file_path} - {size_in_mb:.2f} MB")
            except OSError:
                print(f"Error accessing file: {file_path}")

# Example: Find files larger than 100 MB in the current directory
find_large_files('.', 20)
