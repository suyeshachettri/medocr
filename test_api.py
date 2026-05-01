import requests

def test_prediction(image_path):
    with open(image_path, "rb") as f:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files={"file": f}
        )
    return response.json()

# Test 5 different images
for i in range(1, 6):
    path   = rf"C:\Projects\medocr\data\synthetic\images\synth_0000{i}.png"
    result = test_prediction(path)
    print(f"Image {i}: {result['predicted_text']}")