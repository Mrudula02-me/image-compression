from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte, filters, exposure
from scipy.ndimage import convolve
from PIL import Image
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


# ------------------------------------------
# K-Means from Scratch
# ------------------------------------------
def kmeans_from_scratch(pixels, K, max_iters=25, tol=1e-4):
    np.random.seed(42)
    indices = np.random.choice(pixels.shape[0], K, replace=False)
    centroids = pixels[indices]
    movement_history = []

    for i in range(max_iters):
        distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            pixels[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
            for k in range(K)
        ])

        shift = np.linalg.norm(new_centroids - centroids)
        movement_history.append(shift)

        if shift < tol:
            break

        centroids = new_centroids

    return centroids, labels, movement_history


# ------------------------------------------
# Image Compression Function
# ------------------------------------------
def compress_image(image_path, K):
    orig_size = os.path.getsize(image_path)
    if orig_size < 50 * 1024:
        return None, orig_size, orig_size, 0, None, None, "⚠️ Image too small — cannot compress further."

    image = io.imread(image_path)
    image = image / 255.0
    h, w, c = image.shape

    # Slight blur for smoother clustering
    image_blur = filters.gaussian(image, sigma=0.6, channel_axis=-1)
    pixels = image_blur.reshape(-1, 3)

    # Apply custom K-Means
    centroids, labels, movement_history = kmeans_from_scratch(pixels, K)

    # Reconstruct and enhance
    compressed_pixels = centroids[labels].reshape(h, w, 3)
    blended = 0.85 * compressed_pixels + 0.15 * image

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = np.zeros_like(blended)
    for i in range(3):
        sharpened[..., i] = convolve(blended[..., i], kernel)
    sharpened = np.clip(sharpened, 0, 1)
    final_image = exposure.rescale_intensity(sharpened, in_range=(0, 1))

    # Save output image
    output_filename = f"compressed_{uuid.uuid4().hex[:6]}.jpg"
    output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
    Image.fromarray(img_as_ubyte(final_image)).save(output_path, "JPEG", quality=75, optimize=True)

    comp_size = os.path.getsize(output_path)
    reduction = 100 * (orig_size - comp_size) / orig_size

    if comp_size >= orig_size:
        os.remove(output_path)
        return None, orig_size, comp_size, 0, None, None, "⚠️ Compression ineffective — output larger than input."

    # Plot 1: Centroid movement
    graph_filename = f"movement_{uuid.uuid4().hex[:6]}.png"
    graph_path = os.path.join(app.config['RESULT_FOLDER'], graph_filename)
    plt.figure(figsize=(8, 4))
    plt.plot(movement_history, marker='o')
    plt.title("📉 K-Means Convergence Graph")
    plt.xlabel("Iteration")
    plt.ylabel("Centroid Shift")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    # Plot 2: Color palette
    palette_filename = f"palette_{uuid.uuid4().hex[:6]}.png"
    palette_path = os.path.join(app.config['RESULT_FOLDER'], palette_filename)
    palette = np.zeros((50, K * 50, 3))
    for i in range(K):
        palette[:, i * 50:(i + 1) * 50, :] = centroids[i]
    plt.figure(figsize=(8, 2))
    plt.imshow(palette)
    plt.axis("off")
    plt.title(f"Color Palette (K={K})")
    plt.tight_layout()
    plt.savefig(palette_path)
    plt.close()

    return output_filename, orig_size, comp_size, reduction, graph_filename, palette_filename, "✅ Compression successful!"


# ------------------------------------------
# Routes
# ------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route("/compress", methods=["POST"])
def compress():
    if "image" not in request.files:
        return "No file part", 400

    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400

    k = int(request.form.get("clusters", 16))
    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    compressed_image, orig_size, comp_size, reduction, graph_filename, palette_filename, message = compress_image(upload_path, k)

    return render_template(
        "result.html",
        original_image=filename,
        compressed_image=compressed_image,
        centroid_graph=graph_filename,
        color_palette=palette_filename,
        original_size=orig_size / 1024,
        compressed_size=comp_size / 1024,
        reduction=reduction,
        k=k,
        message=message
    )


if __name__ == '__main__':
    app.run(debug=True)
