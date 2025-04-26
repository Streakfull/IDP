import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to check a single directory for required files


def check_directory(path, required_files):
    missing_count = 0
    missing_directories = []

    # Walk through each directory in the path
    for root, dirs, files in os.walk(path):
        # Check if all required files are present
        if not all(file in files for file in required_files):
            missing_count += 1
            missing_directories.append(root)
    print("Check single dir")
    return missing_count, missing_directories

# Function to check all directories in parallel


def check_directories(paths, required_files):
    missing_count = 0
    missing_directories = []
    print("Check dir")
    # Use ThreadPoolExecutor to parallelize directory checks
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(
            check_directory, path, required_files): path for path in paths}
        for future in tqdm(as_completed(futures), desc="Checking directories", total=len(paths)):
            result_missing_count, result_missing_directories = future.result()
            missing_count += result_missing_count
            missing_directories.extend(result_missing_directories)

    # Print all directories with missing files
    print("\nDirectories with missing files:")
    for directory in missing_directories:
        print(directory)

    return missing_count


def main():
    # Define the dataset paths
    print("Running Main")
    dataset_paths = [
        "./raw_dataset/latest/full/samples",
        "./raw_dataset/latest/full/clipped_text",
        "./raw_dataset/latest/full/clipped_manual_2",
    ]

    # Define the required files
    required_files = ["x1kv1.txt", "x1kv2.txt"]

    # Check directories and count missing ones
    missing_directories_count = check_directories(
        dataset_paths, required_files)

    # Output the result
    print(
        f"Total directories missing required files: {missing_directories_count}")


if __name__ == "__main__":
    main()
