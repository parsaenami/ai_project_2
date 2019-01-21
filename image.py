import cv2
import numpy as np


def show(label, image):
    cv2.imshow(label, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("C:\\Users\\Parsa\\OneDrive\\university\\semester 5\\AI\\FinalProject\\test.jpg")
# img = cv2.resize(img, (500, 500))
show('(1) open isco\'s picture', img)

blue = img.copy()
blue[:, :, 1], blue[:, :, 2] = 0, 0
show('(2) isco in blue filter', blue)

img0 = cv2.imread("C:\\Users\\Parsa\\OneDrive\\university\\semester 5\\AI\\FinalProject\\test.jpg", 0)
# img0 = cv2.resize(img0, (500, 500))
show('(3) gray scale isco', img0)

gauss_gray = cv2.GaussianBlur(img0, (9, 9), 0)
show('(4) smoothed gaussian filtered gray scale isco', gauss_gray)

rows, cols = img.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, cols/2), 90, 1)
dst = cv2.warpAffine(img, M, (rows, cols))
show('(5) rotated isco', dst)

e = cv2.resize(img, (cols // 2, rows))
show('(6) isco has lost wight', e)

edge = cv2.Canny(img, 100, 200)
show('(7) isco\'s outline', edge)

img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img0, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
new_img = img.copy()
markers = cv2.watershed(new_img, markers)
new_img[markers == -1] = [255, 0, 0]
show('(8) isco\'s segmented image', new_img)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
faces = face_cascade.detectMultiScale(img, 1.07, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (20, 0, 200), 2)
show('(9) isco\'s face', img)

video = cv2.VideoCapture("C:\\Users\\Parsa\\OneDrive\\university\\semester 5\\AI\\FinalProject\\drop.avi")
for i in range(5):
    success, image = video.read()
    print(success) if not success else print()
    cv2.imshow('(10) tears after loosing isco', image)
    cv2.waitKey(500)
cv2.destroyAllWindows()
