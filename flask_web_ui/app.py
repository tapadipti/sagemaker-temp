from flask import Flask, render_template, request
import numpy as np
import boto3, json
from PIL import Image

app = Flask(__name__)
sagemaker_runtime_client = boto3.client('runtime.sagemaker')


def pre_process(image_file):
    # Preprocess the image and convert it to the format expected by the model
    # For example, resizing the image to (28, 28) and converting it to a NumPy array

    import numpy as np
    from PIL import Image

    # Load the image
    image = Image.open(image_file)

    # Resize the image to (28, 28)
    image = image.resize((28, 28))

    # Convert the image to grayscale (if it's not already)
    image = image.convert('L')

    # Convert the PIL image to a NumPy array
    image_array = np.array(image)

    # Normalize the pixel values to [0, 1] as the model expects inputs in this range
    image_array = image_array / 255.0

    # Reshape the image to match the input shape of the model (num_samples, height, width, channels)
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image from the form
        image_file = request.files['file']
        
        # Preprocess the image as shown in the previous code snippet
        # (Resize, convert to grayscale, normalize, and reshape)
        image_array = pre_process(image_file)
        image_list = image_array.tolist()

        # Send the preprocessed image to the SageMaker endpoint for prediction
        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName='td-fm-ep-dev',
            Body=json.dumps(image_list),
            ContentType='application/json'
        )

        result = response['Body'].read().decode('utf-8')
        result_json = json.loads(result)
        print(result_json)

        # Process the prediction response and get the predicted class label
        prediction = np.array(result_json['predictions'][0])
        predicted_class = np.argmax(prediction)

        # Map the class label to the actual class name (e.g., 'T-shirt/top', 'Trouser', etc.)
        label_map = {
            0: "T-shirt or top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
            }
        predicted_class_name = label_map[predicted_class]

        return render_template('index.html', prediction=predicted_class_name)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
