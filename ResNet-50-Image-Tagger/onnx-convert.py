import json
import numpy as np
import onnxruntime
from keras.utils import load_img, img_to_array

model = ".\\model.onnx"
path = "C:\\Users\\lucious\\Pictures\\selfTest\\tumblr_e9ec3ee1c5461b2ef48806aa196f16b9_563c2f60_1280.jpg"

image = load_img(path, target_size=(224, 224))
imgResult = img_to_array(image)
imgResult = np.expand_dims(imgResult, axis=0)
imgResult = imgResult / 255.
img = imgResult

data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(input_name)
result = session.run([output_name], {input_name: data})
prediction = int(np.argmax(np.array(result).squeeze(), axis=0))
print(prediction)
print(result)