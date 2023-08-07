from flask import Flask, render_template, request
import numpy as np
import boto3, json
from PIL import Image

app = Flask(__name__)
sagemaker_runtime_client = boto3.client('runtime.sagemaker')
endpoint_name = 'td-fm-ep-dev'


def pre_process(image_file):
    image = Image.open(image_file)

    # Resize, convert to grayscale, convert to np array and convert to array of required shape
    image = image.resize((28, 28))
    image = image.convert('L')
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image from the form
        image_file = request.files['file']
        
        # Preprocess and convert to list
        image_array = pre_process(image_file)
        image_list = image_array.tolist()

        # Call SageMaker endpoint
        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=json.dumps(image_list),
            ContentType='application/json'
        )

        result = response['Body'].read().decode('utf-8')
        result_json = json.loads(result)
        print(result_json)

        prediction = np.array(result_json['predictions'][0])
        predicted_class = np.argmax(prediction)

        # Map the class label to the class name
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
