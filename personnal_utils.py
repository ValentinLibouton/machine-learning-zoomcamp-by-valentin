def gpu_activation():
    from tensorflow.config import experimental as tf_exp
    physical_devices = tf_exp.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf_exp.set_memory_growth(gpu, True)
