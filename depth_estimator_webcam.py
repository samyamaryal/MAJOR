from tensorflow import keras
import tensorflow as tf
import cv2
from keras_depth import DataGenerator as dg #this imports the whole of keras_depth so it isn't good
# depth_model = tensorflow.keras.models.load_model("./model")

cap = cv2.VideoCapture(0)
model = tf.keras.models.load_model("model")
print("Success")
image = cv2.imread("img.jfif")
print(type(image), image.shape)
image = dg(data=image.reset_index(drop="true"), batch_size=32, dim=(256, 256))
cv2.imshow(image)

#prediction_depth = model.predict(image)

'''
while cap.isOpened():
    success, img = cap.read()
    # img = cv2.cvtColor(img)
    cv2.imshow('Image', img)
    
    
    
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        cap.release() 
        cv2.destroyAllWindows()
        break
print(img.shape)

# depth_model.predict()
'''
