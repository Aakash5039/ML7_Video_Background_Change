import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from scipy.ndimage import gaussian_filter
import time
from google.colab.patches import cv2_imshow

class DeepLabModel():
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def _init_(self):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = '/content/drive/MyDrive/models/frozen_inference_graph_dm05.pb'
        with tf.io.gfile.GFile(graph_def, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options), graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            seg_map: Segmentation map of resized_image.
        """

        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return seg_map

model = DeepLabModel()

def imresize(image):
    INPUT_SIZE = 513
    width, height = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    return resized_image

# Load the TensorFlow Hub module for image harmonization
harmonization_module = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

def harmonize_image(content_image, style_image):
    """Harmonizes content image with style image."""
    content_image = np.array(content_image)[np.newaxis, ...]
    style_image = np.array(style_image)[np.newaxis, ...]

    result = harmonization_module(tf.constant(content_image), tf.constant(style_image))[0]
    return result[0].numpy()

# Open Videos using cv2.VideoCapture
vidcap1 = cv2.VideoCapture('/content/drive/MyDrive/test1.mp4')
vidcap2 = cv2.VideoCapture('/content/drive/MyDrive/test.mp4')

# Check if videos opened successfully
if not vidcap1.isOpened() or not vidcap2.isOpened():
    raise RuntimeError("Error opening one or both video files.")

# Get video properties
fps1 = vidcap1.get(cv2.CAP_PROP_FPS)
fps2 = vidcap2.get(cv2.CAP_PROP_FPS)
print(f'Frames per second of video 1 : {fps1}')
print(f'Frames per second of video 2: {fps2}')

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
outvid = cv2.VideoWriter('outvid.avi', fourcc, 10, (1024, 512))

while True:
    start = time.time()

    success1, image1 = vidcap1.read()
    success2, image2 = vidcap2.read()

    if not (success1 and success2):
        print("Failed to read frame from one or both videos.")
        break

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    image1 = imresize(Image.fromarray(image1))
    image2 = imresize(Image.fromarray(image2))

    model_start_time = time.time()

    seg_map_left = model.run(image1)
    seg_map_left[seg_map_left > 0] = 255
    floatmap = seg_map_left.astype(np.float32) / 255.0
    blurmap = gaussian_filter(floatmap, sigma=3)
    cv2.imwrite("blurseg.png", blurmap * 255)

    model_ex_time = time.time() - model_start_time
    print(f"Model execution time: {model_ex_time:.2f} seconds")

    resized_im_left = np.float32(image1)
    resized_im_right = np.float32(image2)

    # Harmonize images
    harmonized_frame = harmonize_image(resized_im_left, resized_im_right)
    harmonized_frame = np.uint8(harmonized_frame)
    cv2.imwrite("harmonized.png", harmonized_frame)

    end = time.time()
    elapsed = end - start
    print(f"Processing time: {elapsed:.2f} seconds")

    # Blending and saving the result
    im_ori = Image.open('harmonized.png')
    im = im_ori.resize((1024, 512), Image.BICUBIC)
    im = np.array(im, dtype=np.float32)
    if im.shape[2] == 4:
        im = im[:, :, 0:3]

    im = im[:, :, ::-1]
    raw = np.uint8(im)

    result = im[..., ::-1]
    cv2.imwrite("after.png", result)

    # Display the image in Colab
    cv2_imshow(result)

    # Save the video
    result_all = np.concatenate((raw, result), axis=1)
    outvid.write(result_all)

# Release resources
vidcap1.release()
vidcap2.release()
outvid.release()
cv2.destroyAllWindows()