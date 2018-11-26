from .cosine_metric_learning import train_app
from .cosine_metric_learning import cosine_inference
import numpy as np
import tensorflow as tf
import time
import pdb

IMAGE_SHAPE = (128,128,3)

input_var = tf.placeholder(tf.uint8, (None, ) + IMAGE_SHAPE) 

image = np.random.randint(0, 255, (1,)+IMAGE_SHAPE )
embedder = cosine_inference.CosineInference()

#embedder.session.run(feature, feed_dict={input_var, image})
for i in range(1000):
    start = time.time()
    embedder.get_features(image)
    print("time to embed was {}".format(time.time() - start))

