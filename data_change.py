"""
Change three json labels.
"""

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

IMAGE_JSON = ['./dataset/label/train_data.json', './dataset/label/test_data.json', './dataset/label/val_data.json']

IMAGE_TEXT = ['./dataset/label/train_data.txt', './dataset/label/test_data.txt', './dataset/label/val_data.txt']

IMAGE_CLASS_NAMES = './dataset/label/class_names.txt'

image_names = []
labels = []
class_names = []

with open('./dataset/label/train_data.json', encoding='utf-8') as f:
    line = f.readline()
    d = json.loads(line)

    for (key, value) in d.items():
        # print ("dict[%s]=" % key,value)
        image_name = key + ".png"
        label = value
        image_names.append(image_name)
        labels.append(label)
    # print (image_names)
    # print (labels)
    f.close()

# print(image_names)
# print(labels)

for label in labels:
	for i in label:
		# print(i)
		if i not in class_names:
			class_names.append(i)

print("the number of tags is :", len(class_names))
print(class_names)

with open(IMAGE_CLASS_NAMES, "w") as f:

	f.write("CLASS_NAMES = [")

	for i in class_names:
		f.write("'")
		f.write(str(i))
		f.write("'")
		f.write(",")

	f.write("]")
	f.write('\n')

	f.close()


for file_index in range(3):

	image_names = []
	labels = []

	with open(IMAGE_JSON[file_index], encoding='utf-8') as f:
	    line = f.readline()
	    d = json.loads(line)

	    for (key, value) in d.items():
	        # print ("dict[%s]=" % key,value)
	        image_name = key + ".png"
	        label = value
	        image_names.append(image_name)
	        labels.append(label)
	    # print (image_names)
	    # print (labels)
	    f.close()

	print(IMAGE_JSON[file_index], ": the number this dataset is ", len(image_names))



	with open(IMAGE_TEXT[file_index], "w") as f:
		for index in range(len(image_names)):
			image_name = image_names[index]
			label = labels[index]

			num_label = []
			for i in range(len(class_names)):
				num_label.append(0)

			for word in label:
				pos = class_names.index(word)
				num_label[pos] = 1

			# print (label)
			# print (num_label)

			f.write(str(image_name))
			f.write(" ")
			for i in num_label:
				f.write(str(i))
				f.write(" ")
			f.write('\n')

		f.close()
		


