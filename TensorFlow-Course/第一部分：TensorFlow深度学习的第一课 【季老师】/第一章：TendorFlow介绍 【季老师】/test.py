import tensorflow as tf

print(tf.__version__)
print(tf.__path__)
print('GPU', tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))
a = tf.constant(1)
b = tf.constant(1)
c = tf.add(a, b)
print(c)
