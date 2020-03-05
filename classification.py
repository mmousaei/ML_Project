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
        self.train_cropped_images_filenames =[]
        self.train_cropped_images =[]
        self.train_cropped_labels =[]
        self.test_cropped_images_filenames =[]
        self.test_cropped_images =[]
        self.test_cropped_labels =[]


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
    def make_cropped_dataset(self, addr, size):
        for i in range(len(self.filenames)):
            image = cv2.imread(addr+"/"+self.filenames[i].rstrip("\n"))
            # print(addr+"/"+self.filenames[i])
            for j in range(len(self.file_annotations[i])):
                x = int(self.file_annotations[i][j][0])
                y = int(self.file_annotations[i][j][1])
                w = int(self.file_annotations[i][j][2])
                h = int(self.file_annotations[i][j][3])
                label = self.file_annotations[i][j][4]
                # print("x = ", x, "y = ", y, "w = ", w, "h = ", h, "label = ", label)
                # print("img shape = ", image.shape)
                crop_img = image[y:y+h, x:x+w, 0]
                resized_image = cv2.resize(crop_img, (size, size))
                # print("crop_img = ", crop_img)
                self.X_data.append(resized_image)
                self.y_data.append(label.rstrip("\n"))
        
    def read_cropped_image_list(self, addr, train_test = True):
        with open(addr) as openfileobject:
            for line in openfileobject:
                values = line.split(" ")
                values[1] = int(values[1].rstrip("\n"))
                if(train_test):
                    self.train_cropped_images_filenames.append(addr[0:27]+values[0][1:])
                    self.train_cropped_labels.append(values[1])
                else:
                    self.test_cropped_images_filenames.append(addr[0:27]+values[0][1:])
                    self.test_cropped_labels.append(values[1])

    def read_all_cropped_image_list(self, addr, folder_names, num_of_files):
        for i in range(len(folder_names)):
            for j in range(num_of_files[i]):
                self.read_cropped_image_list(addr+"/"+folder_names[i]+"/train_list_"+str(j)+".txt", True)
                self.read_cropped_image_list(addr+"/"+folder_names[i]+"/test_list_"+str(j)+".txt", False)
    
    def read_resize_image(self, addr):
        image = cv2.imread(addr)
        resized_image = cv2.resize(image, (28, 28))
        return resized_image 

    def save_dataset(self, name):
        X = np.array(self.X_data)
        np.save(name, X)
    def save_label(self, name):
        y = np.array(self.y_data)
        np.save(name, y)
    def open_dataset(self, name):
        self.X_data = np.load(name)
    def open_label(self, name):
        self.y_data = np.load(name)

    
    # def read_cropped_images(self, addr):
    #     for i in range(len(self.train_cropped_images_filenames)):
    #         # image = cv2.imread(addr+self.filenames[i])
    #         print(addr[0:27]+self.train_cropped_images_filenames[i])

        




if __name__ == '__main__':
    c = Classify()
    

    # # Cropped Version
    # folder_names = ["bird_vs_nonbird", "hawk_vs_crow"]
    # num_of_files = [5, 10]
    # c.read_all_cropped_image_list("../../data/Classify/cropped/image_lists", folder_names, num_of_files)
    # cv2.imshow("image", c.read_resize_image(c.train_cropped_images_filenames[15000]))
    # cv2.waitKey(0)

    # Real Dataset Version

    # c.read_annotations("../../data/Classify/BIRD_sample.txt")
    # c.make_cropped_dataset("../../data/Classify", size=28)
    # print(c.X_data[2].shape)
    # c.save_dataset("data.npy")
    # c.save_label("label.npy")

    c.open_dataset("data.npy")
    c.open_label("label.npy")
    cv2.imshow("image", c.X_data[2])
    cv2.waitKey(0)