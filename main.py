from src.predict import predict_image

image_path = "test.jpg"

result = predict_image(image_path)

print("Prediction:", result)