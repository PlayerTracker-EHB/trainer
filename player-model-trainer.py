from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')

# Load the data configuration
data = 'data.yaml'

# Train the model
model.train(data=data, epochs=100, imgsz=640, batch=16, workers=4)

# Save the trained model
model.save('trained_model.pt')

# Evaluate the model
# Evaluate the Model
results = model.val(data=data)
print(results)

# Plot The Results
results.plot(show=True, save=True, path='results')