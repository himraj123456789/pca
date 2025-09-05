from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import joblib

# ------------------ Flask setup ------------------
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------ Load PCA model ------------------
model_data = joblib.load("pca_face_model.joblib")
scaler = model_data['scaler']
pca = model_data['pca']
class_means = model_data['class_means']

# ------------------ Prediction function ------------------
def predict_face(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (100, 100))
    img_flat = img_resized.flatten().reshape(1, -1)

    img_scaled = scaler.transform(img_flat)
    img_pca = pca.transform(img_scaled)

    # Find closest class
    min_dist = float("inf")
    predicted_class = None
    for cls, mean_vec in class_means.items():
        dist = euclidean_distances(img_pca, mean_vec.reshape(1, -1))[0][0]
        if dist < min_dist:
            min_dist = dist
            predicted_class = cls

    # Reconstruct closest class mean face
    closest_mean_pca = class_means[predicted_class]
    reconstructed = pca.inverse_transform(closest_mean_pca).reshape(100, 100)
    reconstructed = scaler.inverse_transform(reconstructed.flatten().reshape(1, -1)).reshape(100, 100)
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    return predicted_class, img_resized, reconstructed

# ------------------ Flask route ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    predicted_class = None
    uploaded_image = None
    reconstructed_image = None

    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            # Save uploaded file
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            # Predict
            predicted_class, uploaded_arr, reconstructed_arr = predict_face(img_path)

            # Save processed images for display
            uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            cv2.imwrite(uploaded_path, uploaded_arr)
            recon_path = os.path.join(app.config['UPLOAD_FOLDER'], "recon_" + file.filename)
            cv2.imwrite(recon_path, reconstructed_arr)

            uploaded_image = "static/uploads/" + file.filename
            reconstructed_image = "static/uploads/" + "recon_" + file.filename

    return render_template("index.html",
                           predicted_class=predicted_class,
                           uploaded_image=uploaded_image,
                           reconstructed_image=reconstructed_image)

# ------------------ Run Flask ------------------
if __name__ == "__main__":
    app.run(debug=True)
