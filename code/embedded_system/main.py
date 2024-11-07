import cv2
import sobel
import lbp
import histogram

img_sobel     = sobel.apply_sobel('road.jpg')
img_lbp       = lbp.apply_lbp(img_sobel, 8, 1)
img_histogram = histogram.calculate_histogram(img_lbp)
