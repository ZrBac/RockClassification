import datetime
import json
import math
import os

import keras
from keras.layers import Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt



class model_build():
    def __init__(self, sum_class, batch_size, epoch, learning_rate, model_name, train_image_dir, target_size=48):
        self.sum_class = sum_class
        self.target_size = target_size
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.train_image_dir = train_image_dir
        self.model_dir = os.path.join(os.getcwd(), 'output',  'model',
                                      os.path.basename(self.model_name).split('.')[0])
        print(self.model_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def generate_flow(self):
        datagen = ImageDataGenerator(
            rotation_range=30,
            rescale=1. / 600,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=0.1,
            zoom_range=(0.8, 1.2),
        )

        train_flow = datagen.flow_from_directory(
            directory=self.train_image_dir,
            target_size=(self.target_size, self.target_size),
            batch_size=self.batch_size,
            subset='training'
        )
        print(train_flow.class_indices)
        with open(os.path.join(self.model_dir, "{}_class.json".format(self.model_name.split('.')[0])),
                  "w+") as json_file:
            json.dump(train_flow.class_indices, json_file, indent=2, separators=(",", " : "), ensure_ascii=False)
            json_file.close()

        val_flow = datagen.flow_from_directory(
            directory=self.train_image_dir,
            target_size=(self.target_size, self.target_size),
            batch_size=self.batch_size,
            subset='validation'
        )
        return train_flow, val_flow

    def model_train(self):
        # 构建模型
        pre_m = keras.applications.ResNet101V2(include_top=False,
                                               weights='imagenet',
                                               input_shape=(self.target_size, self.target_size, 3),
                                               pooling="avg")
        x = pre_m.output
        x = Dense(2048, activation='relu', name='fc1')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(2048, activation='relu', name='fc2')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(self.sum_class, activation="softmax", name='predictions')(x)
        sqeue = keras.models.Model(inputs=pre_m.input, outputs=x)

        sqeue.summary()
        # 编译
        sqeue.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])

        # 训练
        saveBestModel = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.model_dir, self.model_name),
            monitor='val_accuracy',
            mode='max',
            verbose=0,
            save_best_only='True')

        train_flow, val_flow = self.generate_flow()

        history = sqeue.fit_generator(
            generator=train_flow,
            steps_per_epoch=math.ceil(train_flow.samples / train_flow.batch_size),
            epochs=self.epoch,
            verbose=1,
            callbacks=[saveBestModel],
            validation_data=val_flow,
            validation_steps=math.ceil(val_flow.samples / val_flow.batch_size)
        )

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(history.epoch, history.history['accuracy'], 'g', label='acc')
        ax1.plot(history.epoch, history.history['val_accuracy'], 'r', label='val_acc')
        ax1.legend()
        ax2.plot(history.epoch, history.history['loss'], 'b', label='loss')
        ax2.plot(history.epoch, history.history['val_loss'], 'yellow', label='val_loss')
        ax2.legend()
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("acc", color='g')
        ax2.set_ylabel("loss", color='b')
        plt.savefig(os.path.join(self.model_dir, "{}_acc_loss.png".format(self.model_name.split('.')[0])), format='png')
        plt.show()


if __name__ == '__main__':
    now_date = datetime.datetime.now().strftime('%m_%d_%H_%M')
    sum_class = 6
    batch_size = 128   #1024
    epoch = 100        #1000
    learning_rate = 0.0001
    model_name = "model_{0}_epoch={1}.h5".format(now_date, epoch)
    train_image_dir = r'G:\PyCharmCode\rockclassification\generated training data'
    model_build(
        sum_class=sum_class,
        batch_size=batch_size,
        epoch=epoch,
        learning_rate=learning_rate,
        model_name=model_name,
        train_image_dir=train_image_dir
    ).model_train()
