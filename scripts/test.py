from predict import predict
import cv2

image1 = cv2.imread("./sample_img/sample_dog.jpg")
image2 = cv2.imread("./sample_img/sample_computer.jpg")

print (predict(image1))
print (predict(image2))
