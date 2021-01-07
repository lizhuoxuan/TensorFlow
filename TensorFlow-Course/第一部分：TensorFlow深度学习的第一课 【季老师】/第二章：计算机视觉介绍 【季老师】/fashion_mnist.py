import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# plt.imshow(test_images[0])
# plt.show()
# print(training_labels[0])
# print(training_images[0])

training_images = training_images / 255.0
test_images = test_images / 255.0
# print(type(test_images))
# print(test_images.shape)
# print(test_images[0].shape)
# print(test_images[0].reshape(1,28,28,1).shape)
# 定义一个Sequential网络
# 也可以先model = tf.keras.models.Sequential()
# 再一层层model.add()
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    # 概率一般使用softmax
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
# 打印模型的概况，如果使用summary，第一层必须指定input_shape
model.summary()

model.compile(optimizer=tf.optimizers.Adam(),
              # 对于分类问题，如果是onehot数据，使用CategoricalCrossentropy，现在不是onehot，所以使用SparseCategoricalCrossentropy
              loss=tf.losses.SparseCategoricalCrossentropy(),
              # 结果里打印精度accuracy
              metrics=['accuracy']
              )


# 训练epochs是次数

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(logs.get("loss"))
        print(logs.get("accuracy"))
        if logs.get("loss") < 0.4:
            print("-----------")
            self.model.stop_training = True


myCallback = MyCallback()
model.fit(training_images, training_labels, epochs=5, callbacks=[myCallback])
# 评估
model.evaluate(test_images, test_labels)
# print(result)

# print(model.metrics_names())
# 判断一张图
# 如果仅仅是一张图，需要将参数转化为三维数组
classification = model.predict(test_images[0].reshape(1, 28, 28))
print(np.argmax(classification))
# 多张图
# classifications = model.predict(test_images)
# print(np.argmax(classifications[0]))
# print(test_labels[0])
