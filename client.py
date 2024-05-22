import numpy as np
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="localhost:8001")

image_data = np.fromfile('example_1.png', dtype=np.uint8)
image_data = np.expand_dims(image_data, axis=0)

# Create the input
input_tensors = [grpcclient.InferInput("input_image", image_data.shape, "UINT8")]
input_tensors[0].set_data_from_numpy(image_data)

# Call API
results = client.infer(model_name="sumen", inputs=input_tensors)
results = results.as_numpy('latex_string')
print(results.astype(str))