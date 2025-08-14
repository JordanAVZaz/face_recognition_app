import gradio as gr
import tensorflow as tf
from gradio_webrtc import WebRTC
import onnxruntime as rt
from sklearn.metrics.pairwise import cosine_similarity

"""
refrences:
https://onnxruntime.ai/docs/tutorials/tf-get-started.html <-- processed in a sperate file
https://onnxruntime.ai/docs/api/python/tutorial.html
https://www.gradio.app/docs/gradio/blocks
https://www.gradio.app/guides/object-detection-from-webcam-with-webrtc
https://pypi.org/project/gradio-webrtc/

Idea: use cosign distance to messure within an identity threshold
You could tune it dynamily, its good to know where to start because the training
margin was alittle high, so the threshold should be a little lenient.

Processing one at a time, we embed with onyx IF the cosign distance is larger then the threshold,
we might use a secondary threshold just to make sure we dont over log new identitys,
or even use like a 3 flags system, where we get a 3 consecutive negative identigy to trigger an
embedding into our local runtime
"""

FACEREC_PCOS_MEAN = 0.87437755
FACEREC_NCOS_MEAN = 0.80952394
SPOOFIN_PCOS_MEAN = 0.99685305 
SPOOFIN_NCOS_MEAN = 0.7924088 
FACEREC_PED_MEAN = 0.41533334
FACEREC_NED_MEAN = 0.63670325
SPOOFIN_PED_MEAN = 0.07235691
SPOOFIN_NED_MEAN = 0.88083166 #negatives for lowest threhold
facerec_model = rt.InferenceSession(
    "embedding.onnx", providers=rt.get_available_providers())
# input_name = sess.get_inputs()[0].name
# pred_onx = sess.run(None, {input_name: X_test.astype(numpy.float32)})[0]
# print(pred_onx)
spoofing_model = rt.InferenceSession(
    "embedding.onnx", providers=rt.get_available_providers())

def cosine_similarity_pairs(a, b):
    a = tf.math.l2_normalize(a, axis=0)
    b = tf.math.l2_normalize(b, axis=0)
    return tf.reduce_sum(a * b).numpy()

def euclidean_distance_mean(set_x, set_y):
    distances = tf.norm(set_x - set_y, axis=1)
    return tf.reduce_mean(distances)

#inputn frame, output label
# expecting 2 sliders, 1 for cos and one for alpha
def detection(img, alpha_cos=0,  alpha_ed=0):
    All the checks...
    if spoofing_model.predict(img) inside SPOOFIN_PCOS_MEAN but not crossing SPOOFIN_NCOS_MEAN 

    if spoofing_model.predict(img) < val
    prediction = embedding_model.run(preprocess_image(image))
    

    pass

# app loop
with gr.Blocks() as demo:
    # conficence slider
    image = WebRTC(label="Stream", mode="send-receive", modality="video")
    conf_threshold = gr.Slider(
        label="Confidence Threshold",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        value=0.30,
    )
    #input function
    image.stream(
        fn=detection,
        inputs=[image, conf_threshold],
        outputs=[image], time_limit=10
    )

if __name__ == "__main__":
    demo.launch()

DIM = 64
def preprocess_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, DIM)
    return image
