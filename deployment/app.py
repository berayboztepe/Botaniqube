from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from image_processing import classes, create_cnn_model
import os

app = Flask(__name__)

model = None  # Global model instance

def load_model():
    global model
    params = {"num_layers": 5, "hidden_units": 256, "num_classes": 38, "image_size": [224, 224], "batch_size": 5}
    model = create_cnn_model(params)
    try:
        model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        

print("Current working directory:", os.getcwd())  # Print the current working directory
print("Files in current directory:", os.listdir())  # List files in the current directory

    
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        img_size = [224, 224]  # Adjust as needed
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = classes[int(predicted.item())]
        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()  # Load model at startup
    app.run(debug=True, host='0.0.0.0', port=6000)



# curl -X POST -F "file=@C:\Users\user\Studia\DataScience\Semestr_II\Project_Deep_Learning\data\01_raw\disease_dataset\test\test\TomatoEarlyBlight2.jpg" http://localhost:6000/predict
# {
#   "prediction": "Tomato___Early_blight"
# }