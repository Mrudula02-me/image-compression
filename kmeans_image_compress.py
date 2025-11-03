import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte, filters, exposure
from PIL import Image
from scipy.ndimage import convolve
import os

# ---------------------------
# STEP 1: Load and preprocess the image
# ---------------------------
img_path = r"D:\MINI\Image compression\image.png"

# Check file size before compression
orig_size = os.path.getsize(img_path)
if orig_size < 50 * 1024:  # Less than 50 KB
    print("⚠️ Image is already too small to compress meaningfully.")
    exit()

image = io.imread(img_path)
image = image / 255.0  # Normalize pixel values to [0, 1]

h, w, c = image.shape
print(f"✅ Original Image Shape: {image.shape}")

# Apply a slight Gaussian blur to smooth noisy edges before K-Means
image_blur = filters.gaussian(image, sigma=0.6, channel_axis=-1)

# ---------------------------
# STEP 2: Reshape for clustering
# ---------------------------
pixels = image_blur.reshape(-1, 3)
print(f"Pixels reshaped for clustering: {pixels.shape}")

# ---------------------------
# STEP 3: K-Means from Scratch
# ---------------------------
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
        print(f"Iteration {i+1}/{max_iters} — Centroid Shift: {shift:.6f}")

        if shift < tol:
            print(f"✅ Converged in {i+1} iterations.")
            break

        centroids = new_centroids

    return centroids, labels, movement_history

# Apply K-Means
K = 16
print("🔹 Applying custom K-Means...")
centroids, labels, movement_history = kmeans_from_scratch(pixels, K)
print("✅ K-Means Done!")

# ---------------------------
# STEP 4: Reconstruct and enhance compressed image
# ---------------------------
compressed_pixels = centroids[labels]
compressed_image = compressed_pixels.reshape(h, w, 3)

# Blend to preserve some detail
blended_image = 0.85 * compressed_image + 0.15 * image

# Sharpen
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened = np.zeros_like(blended_image)
for i in range(3):
    sharpened[..., i] = convolve(blended_image[..., i], kernel)
sharpened = np.clip(sharpened, 0, 1)

# Adjust contrast
final_image = exposure.rescale_intensity(sharpened, in_range=(0, 1))

# ---------------------------
# STEP 5: Save the Compressed Image
# ---------------------------
output_path = rf"D:\MINI\Image compression\compressed_{K}_enhanced.jpg"

# Save as JPEG with controlled quality
img_to_save = Image.fromarray(img_as_ubyte(final_image))
img_to_save.save(output_path, "JPEG", quality=70, optimize=True)

print(f"💾 Saved compressed image at: {output_path}")

# ---------------------------
# STEP 6: Compare File Sizes
# ---------------------------
comp_size = os.path.getsize(output_path)
reduction = 100 * (orig_size - comp_size) / orig_size
print(f"📦 Original: {orig_size/1024:.2f} KB")
print(f"📉 Compressed: {comp_size/1024:.2f} KB")
print(f"💡 Reduction: {reduction:.2f}%")

if comp_size > orig_size:
    print("⚠️ Compressed image is larger — skipping save as compression ineffective.")
    os.remove(output_path)
    exit()

# ---------------------------
# STEP 7: Display Original vs Compressed
# ---------------------------
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(final_image)
plt.title(f"Enhanced Compressed Image (K={K})")
plt.axis("off")
plt.show()

# ---------------------------
# STEP 8: Show Color Palette
# ---------------------------
palette = np.zeros((50, K * 50, 3))
for i in range(K):
    palette[:, i * 50:(i + 1) * 50, :] = centroids[i]

plt.figure(figsize=(10, 2))
plt.imshow(palette)
plt.axis("off")
plt.title(f"Color Palette (K={K})")
plt.show()

# ---------------------------
# STEP 9: Plot Centroid Movement
# ---------------------------
plt.figure(figsize=(8, 4))
plt.plot(movement_history, marker='o')
plt.title("📉 K-Means Convergence Graph")
plt.xlabel("Iteration")
plt.ylabel("Centroid Shift")
plt.grid(True)
plt.show()
