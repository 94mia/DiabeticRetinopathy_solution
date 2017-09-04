import cv2
import numpy as np

scale_size = 256
area = scale_size * scale_size
im_list = ['train_images/' + line.strip() + '_' + str(scale_size) + '.png' for line in open('train_images.txt', 'r')]
mean_image = np.zeros((scale_size, scale_size,3), dtype=float)
std_image = np.zeros((scale_size, scale_size,3),dtype=float)

print("compute mean...")
for im_id in range(len(im_list)):
   im = cv2.imread(im_list[im_id]) 
   mean_image += im
mean_image = mean_image/len(im_list)
mean_b = mean_image[:,:,0].sum()/area
mean_g = mean_image[:,:,1].sum()/area
mean_r = mean_image[:,:,2].sum()/area
print(mean_r)
print(mean_g)
print(mean_b)

print("compute std...")
for im_id in range(len(im_list)):
   im = cv2.imread(im_list[im_id])
   std_image[:,:,0] += (im[:,:,0]-mean_b)**2
   std_image[:,:,1] += (im[:,:,1]-mean_g)**2
   std_image[:,:,2] += (im[:,:,2]-mean_r)**2

std_image /= len(im_list)
std_b = (std_image[:,:,0].sum()/area)**(1.0/2)
std_g = (std_image[:,:,1].sum()/area)**(1.0/2)
std_r = (std_image[:,:,2].sum()/area)**(1.0/2)
print(std_r)
print(std_g)
print(std_b)
