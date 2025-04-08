import cv2  # to read and process images
import matplotlib.pyplot as plt  # to show resultant images 
import mediapipe as mp
import os

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose
# Setting up the Pose model for images.
pose_img = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

# Initializing mediapipe drawing class to draw landmarks on specified image.
mp_drawing = mp.solutions.drawing_utils

def estimPose_img(input_file, pose=pose_img, landmarks_c=(234, 63, 247), connection_c=(117, 249, 77), 
                  thickness=2, circle_r=5):
    # Read the input image
    if isinstance(input_file, str):
        input_img = cv2.imread(input_file)
    else:
        input_img = input_file
    
    # Create a copy of the input image
    output_img = input_img.copy()
    
    # Convert the image from BGR into RGB format.
    RGB_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(RGB_img)
    
    # Retrieve the height and width of the input image.
    height, width, _ = input_img.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(output_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=landmarks_c, thickness=thickness, circle_radius=circle_r),
                                  mp_drawing.DrawingSpec(color=connection_c, thickness=thickness, circle_radius=circle_r))
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), landmark.z))
    
    return output_img, landmarks

# Base path for images
base_path = '/ballet_positions/testing/dataset_train'
files = os.listdir(base_path)
IMAGE_FILES = [os.path.join(base_path, f) for f in files if f.lower().endswith(('.jpg', '.jpeg'))]

# Process each image
for num, img_file in enumerate(IMAGE_FILES, start=1):
    im, landmarks = estimPose_img(img_file)
    
    out_path = '/ballet_positions/testing/mediapipe_results'
    output_path = os.path.join(out_path, f"_pose_detected{num}.jpg")
    
    cv2.imwrite(output_path, im)
    print(f"Сохранено изображение с результатами: {output_path}")