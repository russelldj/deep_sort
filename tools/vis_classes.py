import numpy as np
import matplotlib.pyplot as plt

# goal: to take a MOT-formated file and output the frequency of all the classes

DETECTION_FILE = "/home/drussel1/data/readonly/detections_and_descriptors/ADL_18_Mask_with_classes_Cosine_extractor/P_18.npy" 

dets = np.load(DETECTION_FILE)
classes = dets[:,1]
plt.hist(classes,bins=80)
plt.show()
