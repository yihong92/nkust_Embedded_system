import cv2
import sobel
import lbp2


img_sobel = sobel.apply_sobel('road.jpg')
img_lbp   = lbp2.main()
