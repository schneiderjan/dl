import os
import pandas as pd
import numpy as np
from PIL import Image

nr_channels = 3
nr_pixels = 224
image_dim = (nr_pixels, nr_pixels)

data = pd.DataFrame(columns=['label', 'scientific_name', 'common_name', 'image_dir', 'image_name','image'])
plant_info = pd.read_csv('plant_info.csv', sep=';')

print('Getting all leaf images')
directory = "leaves"
leaves = []
for root, dirs, files in os.walk(directory):
    for file in files:
        image_dir = os.path.join(root, file)
        image_nr = file.split('.')[0]
        print(image_dir)
        for ix, row in plant_info.iterrows():
            if row['filename_low'] <= int(image_nr) <= row['filename_high']:
                to_append = {'label': row['label'], 'scientific_name': row['Scientific Name'],
                             'common_name': row['Common Name(s)'], 'image_dir': image_dir, 'image_name': file}
                # print(to_append)
                data = data.append(to_append, ignore_index=True)

print('Resizing images')
# # resize image
# for idx, row in data.iterrows():
#     image = Image.open(data.at[idx, 'image_dir'])
#     image = image.resize(image_dim)
#     image.save(data.at[idx, 'image_dir'])

print('Get all channel values')
vals_r = []
vals_g = []
vals_b = []
# get values per channel and store in df
for idx, row in data.iterrows():
    image = Image.open(data.at[idx, 'image_dir'])
    pixels = np.asarray(image).astype('float32')
    max_pixel = 255
    # scale to 0 - 1
    # data.at[idx, 'image'] = pixels
    pixels = pixels / max_pixel
    data.at[idx, 'image'] = pixels

    for i in range(nr_pixels):
        for j in range(nr_pixels):
            for k in range(nr_channels):
                if k == 0:
                    vals_r.append(pixels[i][j][k])
                elif k == 1:
                    vals_g.append(pixels[i][j][k])
                elif k == 2:
                    vals_b.append(pixels[i][j][k])

print('Calculate channel mean')
mean_r = sum(vals_r) / float(len(vals_r))
mean_g = sum(vals_g) / float(len(vals_g))
mean_b = sum(vals_b) / float(len(vals_b))

print(mean_r)
print(mean_g)
print(mean_b)

print('Pre-process images')
# subtract mean per channel calculated over all images
for idx, row in data.iterrows():
    pixels = data.at[idx, 'image']
    for i in range(nr_pixels):
        for j in range(nr_pixels):
            for k in range(nr_channels):
                if k == 0:
                    pixels[i][j][k] = pixels[i][j][k] - mean_r
                elif k == 1:
                    pixels[i][j][k] = pixels[i][j][k] - mean_g
                elif k == 2:
                    pixels[i][j][k] = pixels[i][j][k] - mean_b
    data.at[idx, 'image'] = pixels


print(data.head())
print(data.tail())
print(data.info())

print('Save to data to pickle')
data.to_pickle('data.pkl')
print('Save complete jaaaaa')
