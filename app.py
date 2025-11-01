from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from PIL import Image
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# ---------------------------
# K-Means from Scratch
# ---------------------------
def kmeans_from_scratch(pixels, K, max_iters=20, tol=1e-4):
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


# ---------------------------
# Compress Image Function
# ---------------------------
def compress_image_kmeans(image_path, K):
    image = io.imread(image_path)
    image = image / 255.0
    h, w, c = image.shape

    pixels = image.reshape(-1, 3)
    centroids, labels, movement_history = kmeans_from_scratch(pixels, K)

    compressed_pixels = centroids[labels]
    compressed_image = compressed_pixels.reshape(h, w, 3)

    # Save compressed image
    output_filename = f"compressed_{uuid.uuid4().hex[:6]}.png"
    output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
    Image.fromarray(img_as_ubyte(compressed_image)).save(output_path)

    # Compare file sizes
    orig_size = os.path.getsize(image_path)
    comp_size = os.path.getsize(output_path)
    reduction = 100 * (orig_size - comp_size) / orig_size

    # Save centroid movement graph
    graph_filename = f"centroid_graph_{uuid.uuid4().hex[:6]}.png"
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

    return output_filename, orig_size, comp_size, reduction, graph_filename


# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compress', methods=['POST'])
def compress():
    if 'image' not in request.files:  # ✅ Changed from 'file'
        return redirect(url_for('index'))
    
    file = request.files['image']  # ✅ Changed from 'file'
    if file.filename == '':
        return redirect(url_for('index'))

    k = int(request.form['clusters'])  # ✅ matches your <select name="clusters">
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    compressed_filename, orig_size, comp_size, reduction, graph_filename = compress_image_kmeans(filepath, k)

    return render_template(
        'result.html',
        original_image=file.filename,
        compressed_image=compressed_filename,
        k=k,
        original_size=orig_size/1024,
        compressed_size=comp_size/1024,
        reduction=reduction,
        centroid_graph=graph_filename
    )


if __name__ == '__main__':
    app.run(debug=True)
