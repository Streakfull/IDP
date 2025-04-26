import os
from tqdm import tqdm


def clean_incomplete_folders(dataset_path, required_files, verbose=True):

    deleted_folders = 0

    # Iterate through each folder in the root dataset directory
    for folder in tqdm(os.listdir(dataset_path), desc="Checking folders"):
        folder_path = os.path.join(dataset_path, folder)

        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue

        # Check if all required files exist in the folder
        if not all(os.path.isfile(os.path.join(folder_path, file)) for file in required_files):
            # Delete the folder if any required file is missing
            if verbose:
                print(f"Deleting incomplete folder: {folder_path}")
            for root, dirs, files in os.walk(folder_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(folder_path)
            deleted_folders += 1

    return deleted_folders


if __name__ == "__main__":
    # Example usage when running this script directly
    dataset_path = "./raw_dataset/frame_pairs_far"
    required_files = ["x1.jpg", "x2.jpg", "x1.txt", "x2.txt"]

    deleted = clean_incomplete_folders(dataset_path, required_files)
    print(f"Total folders deleted: {deleted}")
