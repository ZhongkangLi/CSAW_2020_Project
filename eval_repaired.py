import keras
import h5py
import sys
import tensorflow
import numpy as np

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data/255, y_data

def eval(model, x_test_c, y_test_c, x_test_bd, y_test_bd):
    clean_label_p = np.argmax(model.predict(x_test_c), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, y_test_c))*100
    print('Classification accuracy:', class_accu)
        
    bd_label_p = np.argmax(model.predict(x_test_bd), axis=1)
    asr = np.mean(np.equal(bd_label_p, y_test_bd))*100
    print('Attack Success Rate:', asr)

def main():
    #load data files
    clean_test_data = 'data/clean_test_data.h5'
    sunglass_data = 'data/sunglasses_poisoned_data.h5'
    anonymous_data = 'data/anonymous_1_poisoned_data.h5'
    multi_eyebrows_data = 'data/Multi-trigger Multi-target/eyebrows_poisoned_data.h5'
    multi_lipstick_data = 'data/Multi-trigger Multi-target/lipstick_poisoned_data.h5'
    multi_sunglass_data = 'data/Multi-trigger Multi-target/sunglasses_poisoned_data.h5'

    x_test_clean, y_test_clean = data_loader(clean_test_data)
    x_sunglass, y_sunglass = data_loader(sunglass_data)
    x_anonymous, y_anonymous = data_loader(anonymous_data)
    x_multi_eyebrow, y_multi_eyebrow = data_loader(multi_eyebrows_data)
    x_multi_lipstick, y_multi_lipstick = data_loader(multi_lipstick_data)
    x_multi_sunglass, y_multi_sunglass = data_loader(multi_sunglass_data)

    model_sunglass = 'models/sunglasses_bd_net.h5'
    model_anonymous_1 = 'models/anonymous_1_bd_net.h5'
    model_multi = 'models/multi_trigger_multi_target_bd_net.h5'

    sunglass = keras.models.load_model(model_sunglass)
    anonymous = keras.models.load_model(model_anonymous_1)
    multi = keras.models.load_model(model_multi)

    repaired_sunglass_file = 'repaired_models/repaired_sunglass.h5'
    repaired_anonymous1_file ='repaired_models/repaired_anonymous_1.h5'
    repaired_multi_file = 'repaired_models/repaired_multi.h5'

    repaired_sunglass = keras.models.load_model(repaired_sunglass_file)
    repaired_anonymous = keras.models.load_model(repaired_anonymous1_file)
    repaired_multi = keras.models.load_model(repaired_multi_file)   

    print("For sunglass BdNet, before repairing:")
    eval(sunglass, x_test_clean, y_test_clean, x_sunglass, y_sunglass)
    print("After repairing:")
    eval(repaired_sunglass, x_test_clean, y_test_clean, x_sunglass, y_sunglass)
    print(" ")
    print("For anonymous_1 BdNet, before repairing:")
    eval(anonymous, x_test_clean, y_test_clean, x_anonymous, y_anonymous)  
    print("After repairing:")  
    eval(repaired_anonymous, x_test_clean, y_test_clean, x_anonymous, y_anonymous)    
    print(" ")
    print("For multi-trigger and multi-target BdNet, before repairing:")
    print("For eyebrow poisoned data:")
    eval(multi, x_test_clean, y_test_clean, x_multi_eyebrow, y_multi_eyebrow)
    print("For lipstick poisoned data:")
    eval(multi, x_test_clean, y_test_clean, x_multi_lipstick, y_multi_lipstick)
    print("For sunglass poisoned data:")
    eval(multi, x_test_clean, y_test_clean, x_multi_sunglass, y_multi_sunglass) 
    print("After repairing:") 
    print("For eyebrow poisoned data:")
    eval(repaired_multi, x_test_clean, y_test_clean, x_multi_eyebrow, y_multi_eyebrow)
    print("For lipstick poisoned data:")
    eval(repaired_multi, x_test_clean, y_test_clean, x_multi_lipstick, y_multi_lipstick)
    print("For sunglass poisoned data:")
    eval(repaired_multi, x_test_clean, y_test_clean, x_multi_sunglass, y_multi_sunglass)   

if __name__ == '__main__':
    main()