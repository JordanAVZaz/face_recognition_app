import gradio as gr
import cv2
import face_recognition
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Store known face embeddings
known_embeddings = []
threshold = 0.6  # Cosine similarity threshold for new ID

def process_frame(frame):
    global known_embeddings

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and compute encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        label = "Unknown"

        if known_embeddings:
            similarities = cosine_similarity([encoding], known_embeddings)[0]
            max_sim = np.max(similarities)
            if max_sim < threshold:
                known_embeddings.append(encoding)
                label = f"New ID ({len(known_embeddings)})"
            else:
                label = f"ID {np.argmax(similarities)}"
        else:
            known_embeddings.append(encoding)
            label = "First Face"

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

# Build Gradio Interface
iface = gr.Interface(
    fn=process_frame,
    inputs=gr.Camera(),
    outputs=gr.Image(),
    live=True,
    title="Real-Time Face Recognition via Webcam"
)

iface.launch()
