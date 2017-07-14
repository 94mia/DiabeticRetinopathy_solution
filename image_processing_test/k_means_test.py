# import numpy as np
# import cv2
#
# img = cv2.imread('/Users/zhangweidong03/Code/dr/DiabeticRetinopathy_solution/image_processing_test/C0001273_EX_3_7.png')
# Z = img.reshape((-1,3))
#
# # convert to np.float32
# Z = np.float32(Z)
#
# # define criteria, number of clusters(K) and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 3
# ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
#
# # Now convert back into uint8, and make original image
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((img.shape))
#
# cv2.imwrite('kmeans.png', res2)
#
# # cv2.imshow('res2',res2)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#


from PIL import Image
import numpy as np
import cv2
im=cv2.imread('kmeans.png',0)#读取图像

pil_img = Image.fromarray(im)

pil_img.show()

thresh,im=cv2.threshold(im,65,255,cv2.THRESH_BINARY)#用OTSU自动获取阈值并进行二值化，第一个参数表示图像，第二个表示设置阈值（由于我们用OTSU自动设置，所以这里必须填０），第三个参数表示将超过正常范围的像素设置的值，最后一个传入控制参数。

pil_img = Image.fromarray(im)

pil_img.show()
