import cv2
import os


def create_video_from_frames(base_folder, output_video_path, frame_rate=30):
    frame_folders = sorted(os.listdir(base_folder))

    # Initialize variables
    first_frame = None
    frame_width = 0
    frame_height = 0

    for folder in frame_folders:
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            # Get the image from the folder (assuming one image per folder)
            for image_file in sorted(os.listdir(folder_path)):
                image_path = os.path.join(folder_path, image_file)
                if os.path.isfile(image_path):
                    first_frame = cv2.imread(image_path)
                    if first_frame is not None:
                        frame_height, frame_width, _ = first_frame.shape
                        break
        if first_frame is not None:
            break

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc,
                          frame_rate, (frame_width, frame_height))

    # Read and write each frame to the video
    for folder in frame_folders:
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            for image_file in sorted(os.listdir(folder_path)):
                image_path = os.path.join(folder_path, image_file)
                if os.path.isfile(image_path):
                    frame = cv2.imread(image_path)
                    if frame is not None:
                        out.write(frame)  # Write the frame to the video

    # Release the VideoWriter object
    out.release()
    print(f"Video saved as {output_video_path}")


def create_video_from_sequential_frames(input_directory, output_video_path, fps=30):
    # Get list of all frame files and sort them numerically
    frame_files = [f for f in os.listdir(
        input_directory) if f.endswith('.jpg')]
    frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Read the first frame to get the dimensions (height, width)
    first_frame_path = os.path.join(input_directory, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' for .avi
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps, (width, height))

    # Loop through all the frames and write them to the video
    for frame_file in frame_files:
        frame_path = os.path.join(input_directory, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
