import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import struct


class Classify():
    def __init__(self, ):
        self.npy_data = []
        self.filenames = []
        self.file_annotations = []
        self.X_data = []
        self.y_data = []

    # def read_video(self, addr):
    #     cap = cv2.VideoCapture(addr)
    #     ret, frame = cap.read()    
    #     plt.figure()
    #     plt.imshow(frame)

    def read_npy(self, addr):
        self.npy_data = np.load(addr)

    def read_annotations(self, addr):
        with open(addr) as openfileobject:
            tmp = []
            for line in openfileobject:
                if( line[0] == 'I'):
                    self.filenames.append(line)
                    if(len(tmp) != 0):
                        self.file_annotations.append(tmp)
                    tmp = []
                else:
                    values = line.split(",")
                    tmp.append(values)
            self.file_annotations.append(tmp)
    def make_cropped_dataset(self, addr):
        for i in range(len(self.filenames)):
            image = cv2.imread(addr+"/"+self.filenames[i].rstrip("\n"))
            print(addr+"/"+self.filenames[i])
            for j in range(len(self.file_annotations[i])):
                x = int(self.file_annotations[i][j][0])
                y = int(self.file_annotations[i][j][1])
                w = int(self.file_annotations[i][j][2])
                h = int(self.file_annotations[i][j][3])
                label = self.file_annotations[i][j][4]
                print("x = ", x, "y = ", y, "w = ", w, "h = ", h, "label = ", label)
                print("img shape = ", image.shape)
                crop_img = image[y:y+h, x:x+w, 0]
                print("crop_img = ", crop_img)
                self.X_data.append(crop_img)
                self.y_data.append(label.rstrip("\n"))
        


        




if __name__ == '__main__':
    c = Classify()
    # c.read_npy("../data/train_real_images.npy")
    c.read_annotations("../data/Classify/BIRD_v210_1.txt")
    print(c.file_annotations[0][0][0])
    print(len(c.filenames))
    c.make_cropped_dataset("../data/Classify")
    print(len(c.y_data))
    print(c.y_data[0])
    print(c.X_data[0])
    cv2.imshow("image", c.X_data[1])
    cv2.waitKey(0)

