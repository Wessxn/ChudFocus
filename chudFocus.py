import cv2
import mediapipe as mp
from deepface import DeepFace
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles

latest_result = None
detector = 'mediapipe'
MODEL_PATH = 'ChudFocus\\face_landmarker.task' 


def print_result(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result 
    latest_result = result 

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result
)

cap = cv2.VideoCapture(0)
start_time = cv2.getTickCount()

with vision.FaceLandmarker.create_from_options(options) as landmarker:        
    while True: 
        _, frame = cap.read()
        frame_timestamp_ms = int((cv2.getTickCount() - start_time) * 1000 / cv2.getTickFrequency())
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        annotated_image = frame.copy()
    
        if latest_result and latest_result.face_landmarks:
            for face_landmarks in latest_result.face_landmarks:        
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                result = DeepFace.analyze (img_path=annotated_image, actions=['emotion'])
                print(result)

        cv2.imshow('Annotated Face', annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        elif cv2.waitKey(1) & 0xFF == ord('w'):
             cv2.imwrite("MMogged.jpg", annotated_image)

cap.release()
cv2.destroyAllWindows()