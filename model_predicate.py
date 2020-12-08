import json
import os

import cv2
import gdal
import imageio
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from tqdm import tqdm


class predict_image():
    def __init__(self, model_path, model_class_path, image_path, label_class, true_label_path, input_size=48):
        self.model_path = model_path
        self.model_class_path = model_class_path
        self.image_path = image_path
        self.input_size = input_size
        self.label_class = label_class
        self.true_label_path = true_label_path
        self.model_name = os.path.basename(self.model_path).split('.')[0]
        print(self.model_name)
        self.pre_dir = os.path.join(os.getcwd(), 'output', 'predict', self.model_name)
        self.XSize = None  
        self.YSize = None  
        self.dataset = None
        self.dataset, self.XSize, self.YSize = self.get_band(self.image_path)

        if not os.path.exists(self.pre_dir):
            os.makedirs(self.pre_dir)

    def get_band(self, path):
        dataset = gdal.Open(path)
        XSize = dataset.RasterXSize
        YSize = dataset.RasterYSize
        return dataset, XSize, YSize

    def read_all_split(self, dataset, input_size, band_combination):
        while True:
            dataset_mat = dataset.ReadAsArray(0, 0, self.XSize, self.YSize)
            all_tif = [dataset_mat[i - 1] for i in band_combination]
            all_tif = np.array(all_tif, dtype=np.uint8)
            all_tif = np.moveaxis(all_tif, 0, 2)
            padding = cv2.copyMakeBorder(all_tif,
                                         int(input_size / 2),
                                         int(input_size / 2),
                                         int(input_size / 2),
                                         int(input_size / 2),
                                         cv2.BORDER_CONSTANT,
                                         value=0
                                         )
            datagen = ImageDataGenerator(
                rescale=1. / 600,
            )
            all_tif_tmp = datagen.flow(x=np.expand_dims(padding, 0), batch_size=1, shuffle=False)
            padding = all_tif_tmp.__next__()

            for i in range(input_size // 2, input_size // 2 + self.YSize):
                for j in range(input_size // 2, input_size // 2 + self.XSize):
                    if input_size % 2 == 0:
                        temp = padding[:, i - input_size // 2:i + input_size // 2,
                               j - input_size // 2:j + input_size // 2, :]
                    else:
                        temp = padding[:, i - input_size // 2:i + input_size // 2 + 1,
                               j - input_size // 2:j + input_size // 2 + 1, :]

                    yield temp

    def predict(self):
        model = load_model(self.model_path)
        band_combination = [3, 2, 1]
        image_chip = self.read_all_split(dataset=self.dataset, input_size=self.input_size,
                                         band_combination=band_combination)
        predict_label = []
        count = self.XSize * self.YSize
        batch = 1024
        for start in tqdm(range(0, count, batch)):
            end = start + batch if start < (count // batch) * batch else count
            image_data = []
            for index in range(start, end):
                temp_chip = image_chip.__next__()
                image_data.append(np.squeeze(temp_chip))
            pre_index = model.predict_on_batch(np.array(image_data))
            predict_label.extend(np.argmax(pre_index, axis=1))

        np.savetxt(os.path.join(self.pre_dir, '{}_predicated_label.txt'.format(self.model_name)),
                   np.array(predict_label), fmt='%s')

        print('predicted_label_instance：', predict_label[:5])

        return predict_label

    def txt2tif(self, label_data):
        res = np.array(label_data, dtype=np.uint8).reshape((self.YSize, self.XSize))
        imageio.imwrite(os.path.join(self.pre_dir, '{}_predicated_image.tif'.format(self.model_name)), res)

    def metrics(self, label_data):
        model_class = json.load(open(self.model_class_path))
        model_class_reverse = dict((v, k) for k, v in model_class.items())
        label_pre_str = [model_class_reverse[i] for i in label_data]
        label_pre = [self.label_class[k] for k in label_pre_str]

        data_origin = gdal.Open(self.true_label_path)
        label_tru = np.squeeze(
            (data_origin.ReadAsArray(0, 0, data_origin.RasterXSize, data_origin.RasterYSize)).reshape(-1, 1))

        matrix = metrics.confusion_matrix(label_tru, label_pre)
        print('confusion matrix：\n', matrix)
        matrix_norm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]  
        matrix_norm = np.around(matrix_norm, decimals=2)
        print('normalized confusion matrix：\n', matrix_norm)

        precision = precision_score(label_tru, label_pre, average='macro')
        recall = recall_score(label_tru, label_pre, average='macro')
        f1 = f1_score(label_tru, label_pre, average='macro')
        kappa = cohen_kappa_score(label_tru, label_pre)  
        acc = accuracy_score(label_tru, label_pre)  

        metrics_dic = {
            'confusion_matrix_norm': matrix_norm,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'kappa': kappa,
            'acc': acc
        }
        np.save(os.path.join(self.pre_dir, "{}_metrics.npy".format(self.model_name)), metrics_dic, allow_pickle=True)
        print('test image metrics:\nrecall为:{}\nprecision为:{}\nf1为：{}\nkappa为：{}\nacc为：{}'.format(recall, precision, f1, kappa, acc))


if __name__ == '__main__':
    image_path = r'G:\PyCharmCode\rockclassification\image\image.tif'
    model_path = r'G:\PyCharmCode\rockclassification\output\model\model_12_03\model_12_03.h5'
    model_class_path = r'G:\PyCharmCode\rockclassification\output\model\model_12_03\model_12_03.json'
    true_label_class = {
        "板岩": 0,
        "花岗闪长岩": 1,
        "黄土": 2,
        "片岩": 3,
        "水体": 4,
        "松散堆积": 5,

    }
    true_label_path = r'G:\PyCharmCode\rockclassification\image\label.tif'

    pre = predict_image(
        model_path=model_path,
        model_class_path=model_class_path,
        image_path=image_path,
        label_class=true_label_class,
        true_label_path=true_label_path
    )
    pre_label_data = pre.predict()
    pre.txt2tif(pre_label_data)
    pre.metrics(pre_label_data)
