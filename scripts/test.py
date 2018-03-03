from predict import predict
import cv2

# all the sample images which have humans
image1 = cv2.imread("./sample_img/sample_office.jpg")
image2 = cv2.imread("./sample_img/sample_person.jpg")

print (predict(image1))
print (predict(image2))
