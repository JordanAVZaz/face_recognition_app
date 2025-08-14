import gradio as gr
import numpy as np
import onnxruntime as rt
import cv2
import uuid
from sklearn.metrics.pairwise import cosine_similarity

SENSITIVITY = 0.05
SPOOF_SENSITIVITY = .95
SPOOFIN_NCOS_MEAN = 0.7924088 - SPOOF_SENSITIVITY #its too sensative

# my model embeded into the program
facerec = rt.InferenceSession("embedding.onnx", providers=rt.get_available_providers())
facerec_key = facerec.get_inputs()[0].name

#spoof recognition model
spoof_detect = rt.InferenceSession("embedding_spoof.onnx", providers=rt.get_available_providers())
spoof_key = spoof_detect.get_inputs()[0].name

#cached image dictionary
gallery = {}

# function for preprocessing the image, resizing and normalising
def preprocess_image(img_np: np.ndarray) -> np.ndarray:
    small = cv2.resize(img_np, (64,64))
    return (small.astype(np.float32) / 255.0)

# generic embedding function for the models 
# model take np array format, with thes session and its key inputed
def embed(session, key, imgs: np.ndarray) -> np.ndarray:
    return session.run(None, {key: imgs})[0]

# Computes the mean cosine similarity accross the labelled images, this will serve 
# as a base mean to see if the compared image is within a normal range
# This is used as a mean to compare a sbject
# if its outside of the range it is definetely not the subject
def compute_label_mean(label: str) -> float:
    embs = [e for (_, e) in gallery[label].values()]
    if len(embs) < 2:
        return 1.0
    sims = []
    for i in range(len(embs)):
        for j in range(i+1, len(embs)):
            sims.append(cosine_similarity(
                embs[i].reshape(1,-1),
                embs[j].reshape(1,-1)
            )[0,0])
    return float(np.mean(sims))

# finding the closest cosign value in the embedding gallery
# th the mean sim is used to find is something is within an acceptiable range,
# anything outside of that is either not a match and could potentialy be a spoof
def find_best_match(query_emb: np.ndarray, exclude_key=None):
    best = (None, -1.0, None, None)
    for label, entries in gallery.items():
        mean_sim = compute_label_mean(label)
        low, high = mean_sim - SENSITIVITY, mean_sim + SENSITIVITY

        for key, (img, emb) in entries.items():
            if key == exclude_key:
                continue
            sim = cosine_similarity(query_emb.reshape(1,-1), emb.reshape(1,-1))[0,0]
            if low <= sim <= high and sim > best[1]:
                best = (label, sim, key, img)
    return best 

# processes image, puts it in libarary, and returns display values
# aswell as labels the images
def handle_upload(uploaded_img):
    pre = preprocess_image(uploaded_img)
    batch = np.expand_dims(pre, 0) 

    emb_spoof = embed(spoof_detect, spoof_key, batch)[0]
    sim_sp = cosine_similarity(emb_spoof.reshape(1,-1), np.zeros((1,emb_spoof.size)))[0,0]
    if sim_sp < SPOOFIN_NCOS_MEAN:
        out = uploaded_img.copy()
        cv2.putText(
            out, "!SPOOF!", 
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
            1.2, (0, 0, 255), 3
        )
        return uploaded_img, out, "!SPOOF!"

    emb_rec = embed(facerec, facerec_key, batch)[0] 
    qkey = str(uuid.uuid4())
    label, sim, key, match_img = find_best_match(emb_rec, exclude_key=None)

    if label:
        result = f"Matched: {label} (sim={sim:.2f})"
        out_img = match_img
        gallery[label][qkey] = (uploaded_img, emb_rec)
    else:
        new_label = f"id_{len(gallery)}"
        gallery[new_label] = {qkey: (uploaded_img, emb_rec)}
        result = f"Enrolled new ID: {new_label}"
        out_img = uploaded_img

    return uploaded_img, out_img, result

# main loop
# the page itself
with gr.Blocks() as demo:
    gr.Markdown("## Face + Spoof Lookup (with static‐negative threshold)")
    #Uploader object native to gradio, creates a numpy value
    uploader = gr.Image(type="numpy", label="Upload a face image")
    btn      = gr.Button("Process")
    qview    = gr.Image(label="Query")
    # Macthed: Matched with something in the libabry
    # enrolled: uploaded to the library with no match
    # spoof: spoof detected image
    mview    = gr.Image(label="Matched/Enrolled/Spoof")
    lbl      = gr.Textbox(label="Result")
    #defines clickables
    btn.click(
        fn=handle_upload,
        inputs=[uploader],
        outputs=[qview, mview, lbl]
    )

if __name__ == "__main__":
    demo.launch()
