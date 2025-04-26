import os
from tqdm import tqdm

# Function to check directories for required files


def check_directories(paths, required_files):
    missing_count = 0
    missing_directories = []

    for path in paths:
        # Walk through each directory in the path
        for root, dirs, files in tqdm(os.walk(path), desc=f"Checking {path}"):
            # Check if all required files are present
            if not all(file in files for file in required_files):
                missing_count += 1
                missing_directories.append(root)

    # Print all directories with missing files
    print("\nDirectories with missing files:")
    for directory in missing_directories:
        print(directory)

    return missing_count


def main():
    # Define the dataset paths
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
