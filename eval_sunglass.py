import keras
import h5py
import sys
import tensorflow
from PIL import Image
import numpy as np

repaired_model_filename = str(sys.argv[1])
img_filename = str(sys.argv[2])
bd_model_filename = 'models/sunglasses_bd_net.h5'
def main():
    
    image = np.expand_dims(np.asarray(Image.open(img_filename)), axis=0)/255

    repaired_model = keras.models.load_model(repaired_model_filename)
    
    pred_repaired = repaired_model.predict(image)
    bd_model = keras.models.load_model(bd_model_filename)
    bd_repaired = bd_model.predict(image)

    if pred_repaired == bd_repaired:
        print("Class:", repaired_model)
    else:
        print("Class: 1283 (poisoned image)")

if __name__ == '__main__':
    main()