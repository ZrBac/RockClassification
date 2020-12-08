import os
import sys

import cv2
import gdal
import imageio
import numpy as np
import ogr
import pandas as pd


class preprocessing():
    def __init__(self, points_dir, image_path, chip_size=48):
        self.points_dir = points_dir
        self.image_path = image_path
        self.chip_SIZE = chip_size
        self.image_chip_dir = os.path.join(os.getcwd(), 'generated training data')

        self.dataset, self.pic_cols, self.pic_rows = self.get_band(self.image_path)

    def get_band(self, path):
        dataset = gdal.Open(path)
        XSize = dataset.RasterXSize
        YSize = dataset.RasterYSize
        return dataset, XSize, YSize

    def shp_to_csv(self):
        filelist = os.listdir(self.points_dir)
        merge_data = np.array(['x', 'y', 'class'])
        for file in filelist:
            if file.split('.')[-1] == 'shp':
                label = file.split('.')[0]
                abs_file = os.path.join(self.points_dir, file)
                driver = ogr.GetDriverByName('ESRI Shapefile')
                ds = driver.Open(abs_file, 0)
                if ds is None:
                    print("Could not open", abs_file)
                    sys.exit(1)
                layer = ds.GetLayer()

                numFeatures = layer.GetFeatureCount()
                extent = layer.GetExtent()
                feature = layer.GetNextFeature()
                xs = []
                ys = []
                labels = []
                while feature:
                    geometry = feature.GetGeometryRef()
                    x = geometry.GetX()
                    y = geometry.GetY()
                    xs.append(x)
                    ys.append(y)
                    labels.append(label)
                    feature = layer.GetNextFeature()
                data = [xs, ys, labels]
                data = np.array(data)
                data = data.transpose()
                merge_data = np.vstack((merge_data, data))
        return np.array(merge_data)

    def gen_image(self):
        print('--------genreating training data--------')
        data_csv = pd.DataFrame(data=self.shp_to_csv(), columns=['X', 'Y', 'class'])
        data_csv.drop(index=0, inplace=True)

        data_csv[['X', 'Y']] = data_csv[['X', 'Y']].astype(float)
        data_csv[['class']] = data_csv[['class']].astype(str)

        rows = len(data_csv)
        cols = data_csv.columns.size

        geo = self.dataset.GetGeoTransform()
        pic_LEFT = geo[0]
        pic_UP = geo[3]
        xPIXEL = geo[1]
        yPIXEL = geo[5]
        pic_DOWN = pic_UP + yPIXEL * self.pic_rows
        pic_RIGHT = pic_LEFT + xPIXEL * self.pic_cols

        data_csv['x'] = round((data_csv['X'] - pic_LEFT) / xPIXEL)
        data_csv['y'] = round((data_csv['Y'] - pic_UP) / yPIXEL)

        band3 = self.dataset.GetRasterBand(3)
        band2 = self.dataset.GetRasterBand(2)
        band1 = self.dataset.GetRasterBand(1)

        x = data_csv['x'] - (self.chip_SIZE) / 2
        y = data_csv['y'] - (self.chip_SIZE) / 2
        X = data_csv['X']
        label = data_csv['class']
        count = 1
        image_count = 1
        for index, (i, j, label, X) in enumerate(zip(x, y, label, X)):
            i = float(i)
            j = float(j)

            r = band3.ReadAsArray(i, j, self.chip_SIZE, self.chip_SIZE)
            g = band2.ReadAsArray(i, j, self.chip_SIZE, self.chip_SIZE)
            b = band1.ReadAsArray(i, j, self.chip_SIZE, self.chip_SIZE)

            output_dir = os.path.join(self.image_chip_dir, '{}'.format(label))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            try:
                img2 = cv2.merge(np.uint8([r, g, b]))
            except TypeError:
                pass
            else:

                imageio.imwrite(os.path.join(output_dir, '{}.tif'.format(image_count)),
                                img2)
                image_count += 1
            count = count + 1

        print('genreate {} imgs, total repate {} times。'.format((image_count - 1), count - 1))
        return 0


if __name__ == '__main__':
    points_dir = r"G:\PyCharmCode\rockclassification\image\points"
    image_path = r"G:\PyCharmCode\rockclassification\image\image.tif"

    pre = preprocessing(
        points_dir=points_dir,
        image_path=image_path,
        chip_size=48
    )
    pre.gen_image()
