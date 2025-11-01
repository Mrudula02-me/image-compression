import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from PIL import Image
import os

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


def compress_image(img_path, K, output_folder="static/outputs"):
    image = io.imread(img_path)
    image = image / 255.0
    h, w, c = image.shape
    pixels = image.reshape(-1, 3)

    # Run our custom K-Means
    centroids, labels, movement_history = kmeans_from_scratch(pixels, K)

    # Reconstruct image
    compressed_pixels = centroids[labels]
    compressed_image = compressed_pixels.reshape(h, w, 3)

    # Save compressed image
    os.makedirs(output_folder, exist_ok=True)
    output_image_path = os.path.join(output_folder, f"compressed_{K}_colors.png")
    Image.fromarray(img_as_ubyte(compressed_image)).save(output_image_path)

    # Save centroid movement graph
    plt.figure(figsize=(8, 4))
    plt.plot(movement_history, marker='o', color='cyan')
    plt.title("Centroid Movement (Convergence Graph)")
    plt.xlabel("Iteration")
    plt.ylabel("Centroid Shift")
    plt.grid(True)
    centroid_plot_path = os.path.join(output_folder, f"centroid_movement_{K}.png")
    plt.savefig(centroid_plot_path, bbox_inches='tight')
    plt.close()

    # Compare file sizes
    orig_size = os.path.getsize(img_path)
    comp_size = os.path.getsize(output_image_path)
    reduction = 100 * (orig_size - comp_size) / orig_size

    return output_image_path, centroid_plot_path, orig_size, comp_size, reduction
