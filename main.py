import dlib
import cv2
import dlib
from PIL import Image, ImageTk
import random
import numpy as np
from tkinter import Tk, Label, StringVar, font
import os

# Initialize the face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("src/shape_predictor_68_face_landmarks.dat")
video_capture = cv2.VideoCapture(0)

IMG_PATH = "god.png"
ALT_IMG_PATH = "god_save.png"

PHOTO_SIZE = (800,800)
FACE_PADDING = 0.5

last_landmarks = None

LEFT_EYE_LEFT_BOUND = 120
LEFT_EYE_RIGHT_BOUND = 370
RIGHT_EYE_LEFT_BOUND = 270
RIGHT_EYE_RIGHT_BOUND = 520

FACE_HEIGHT_BOUND = 100

def is_face_in_bounds(landmarks):
    global LEFT_EYE_LEFT_BOUND
    global LEFT_EYE_RIGHT_BOUND
    global RIGHT_EYE_LEFT_BOUND
    global RIGHT_EYE_RIGHT_BOUND

    global FACE_HEIGHT_BOUND

    # if eyes are outside of vertical bounds
    if landmarks[36][0] < LEFT_EYE_LEFT_BOUND or landmarks[36][0] > LEFT_EYE_RIGHT_BOUND:
        return False
    if landmarks[39][0] < LEFT_EYE_LEFT_BOUND or landmarks[39][0] > LEFT_EYE_RIGHT_BOUND:
        return False
    if landmarks[42][0] < RIGHT_EYE_LEFT_BOUND or landmarks[42][0] > RIGHT_EYE_RIGHT_BOUND:
        return False
    if landmarks[45][0] < RIGHT_EYE_LEFT_BOUND or landmarks[45][0] > RIGHT_EYE_RIGHT_BOUND:
        return False
    
    # if face is too small
    ld = landmarks[8][1] - landmarks[19][1]
    rd = landmarks[8][1] - landmarks[24][1]
    if min(ld,rd) < FACE_HEIGHT_BOUND:
        return False

    return True

def align_face_and_get_landmarks(image, rgb_image, face):
    
    global PHOTO_SIZE
    global FACE_PADDING
    # Get facial landmarks
    shape = predictor(image, face)
    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

    if not is_face_in_bounds(landmarks):
        return None,None

    # Align the face
    left_eye = np.mean(landmarks[36:42], axis=0)  # Average of left eye points
    right_eye = np.mean(landmarks[42:48], axis=0)  # Average of right eye points

    # Calculate eye center and angle between eyes
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Define transformation matrix for rotation
    M = cv2.getRotationMatrix2D(eye_center, angle, 1)

    # Apply affine transformation to align the image
    aligned_image = cv2.warpAffine(rgb_image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    # Transform landmarks to the aligned image
    ones = np.ones((len(landmarks), 1))
    landmarks_homo = np.hstack([landmarks, ones])  # Convert to homogeneous coordinates
    aligned_landmarks = np.dot(M, landmarks_homo.T).T  # Apply transformation matrix

    # Crop the face region with padding
    x_min, y_min = np.min(aligned_landmarks, axis=0)
    x_max, y_max = np.max(aligned_landmarks, axis=0)
    width = x_max - x_min
    height = y_max - y_min

    # Add padding
    x_min = max(0, int(x_min - FACE_PADDING * width))
    x_max = min(aligned_image.shape[1], int(x_max + FACE_PADDING * width))
    y_min = max(0, int(y_min - FACE_PADDING * height))
    y_max = min(aligned_image.shape[0], int(y_max + FACE_PADDING * height))

    # Crop the aligned face
    cropped_face = aligned_image[y_min:y_max, x_min:x_max]

    # Resize the cropped face to the output size
    aligned_resized_image = cv2.resize(cropped_face, PHOTO_SIZE, interpolation=cv2.INTER_AREA)

    # Resize landmarks to match the resized image
    scale_x = PHOTO_SIZE[0] / (x_max - x_min)
    scale_y = PHOTO_SIZE[1] / (y_max - y_min)
    resized_landmarks = np.array([
        [(x - x_min) * scale_x, (y - y_min) * scale_y] for (x, y) in aligned_landmarks
    ])

    left_eye = np.mean(resized_landmarks[36:42], axis=0)
    right_eye = np.mean(resized_landmarks[42:48], axis=0)

    return resized_landmarks, aligned_resized_image

def are_landmarks_same(landmarks1, landmarks2, threshold=50):
    if landmarks1.shape != landmarks2.shape:
        print("Landmark shapes do not match.")
        return False

    # Calculate Euclidean distance between corresponding points
    distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)

    # Check if all distances are within the threshold
    if np.all(distances <= threshold):
        return True
    else:
        return False

def capture_photo():
    global last_landmarks

    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture video frame. Exiting.")
        return

    # Convert to grayscale for Dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    if len(faces) == 0:
        return

    face = faces[0]
    
    """x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Face Detection", frame)"""

    resized_landmarks, aligned_img = align_face_and_get_landmarks(gray, frame, face)

    if resized_landmarks is None and aligned_img is None:
        return None

    if last_landmarks is not None and are_landmarks_same(resized_landmarks, last_landmarks):
        return None

    last_landmarks = resized_landmarks

    """for i in range(0,68):
        point = tuple(resized_landmarks[i].astype(int)) 
        cv2.circle(aligned_img, point, 5, (255, 255, 255), 10)
    """

    return Image.fromarray(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))

def merge_photo(old_image, new_image):
    global PHOTO_SIZE
    if new_image is None:
        return None

    rows = 10
    cols = 10

    width, height = PHOTO_SIZE
    cell_width = width // cols
    cell_height = height // rows

    merged_image = old_image
    for row in range(rows):
        for col in range(cols):
            x0 = col * cell_width
            y0 = row * cell_height
            x1 = x0 + cell_width
            y1 = y0 + cell_height

            if random.choice([True, False]) :
                merged_image.paste(new_image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1))


    return merged_image


def init_photo():

    global IMG_PATH
    global ALT_IMG_PATH
    old_image = None
    try:
        old_image = Image.open(IMG_PATH)
    except Exception as e:
        alt_img_path = ALT_IMG_PATH
        old_image = Image.open(alt_img_path)

    return old_image

def take_and_merge(old_image):
    global IMG_PATH

    new_image = capture_photo()
    merged_image = merge_photo(old_image, new_image)
    if merged_image is not None:
        merged_image.save(IMG_PATH)

def update_image(label):
    try:
        old_image = Image.open(IMG_PATH)
    except Exception as e:
        alt_img_path = ALT_IMG_PATH
        old_image = Image.open(alt_img_path)

    img = ImageTk.PhotoImage(old_image)
    label.config(image=img)
    label.image = img

def main():
    global IMG_PATH
    root = Tk()
    root.title('Meet Your God')
    root.attributes('-fullscreen', True)
    root.configure(background='black')

    label = Label(root, background='black')
    label.pack(expand=True)

    original_photo = init_photo()

    def periodic_task():

        updated_photo = take_and_merge(original_photo)
        update_image(label)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            print("Exiting program.")
            root.quit()  # Correctly exit the Tkinter loop
            return
        
        root.after(100, periodic_task)

    if os.path.exists(IMG_PATH):
        update_image(label)

    root.after(0, periodic_task)

    root.mainloop()



if __name__ == '__main__':
    main()
