import cv2
import numpy as np
from skimage.feature import hog
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score  # Import silhouette_score

# Function to load and resize images from a folder
def load_and_resize_images_from_folder(folder_path, size=(128, 128)):  # Increased image size
    images = []
    filenames = []
    
    # List all image files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # You can add other formats if needed
            img_path = os.path.join(folder_path, filename)
            
            # Load the image
            img = cv2.imread(img_path)
            
            # Resize the image to avoid small image size issues
            img_resized = cv2.resize(img, size)
            
            # Convert to grayscale (HOG operates on single channel)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            # Append to the list of images and their filenames
            images.append(img_gray)
            filenames.append(filename)
    
    return images, filenames

# Function to extract HOG features from images
def extract_hog_features(images):
    hog_features = []
    hog_visualizations = []  # To store HOG visualizations
    for img in images:
        # Compute the HOG features for the image and get the HOG visualization
        features, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_features.append(features)
        hog_visualizations.append(hog_image)
        
    return np.array(hog_features), hog_visualizations

# Function to apply PCA for dimensionality reduction
def apply_pca(hog_features, n_components=None):
    # Standardize the HOG features before PCA (important for clustering)
    scaler = StandardScaler()
    hog_features_scaled = scaler.fit_transform(hog_features)

    # Determine the number of components dynamically if not provided
    if n_components is None:
        n_components = min(hog_features_scaled.shape[0], hog_features_scaled.shape[1])  # min(#samples, #features)
    
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(hog_features_scaled)
    
    # Optionally print the explained variance to check how much information is retained
    print(f"Explained variance ratio: {np.cumsum(pca.explained_variance_ratio_)}")
    
    return pca_features, scaler, pca

# Apply KMeans clustering to the extracted HOG features
def apply_kmeans_clustering(hog_features, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300, init='k-means++')
    kmeans.fit(hog_features)
    return kmeans

# Function to visualize the HOG features along with images
def visualize_hog_for_test_image(test_image, hog_image):
    # Display HOG features for the test image
    plt.figure(figsize=(12, 6))
    
    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(test_image, cmap='gray')
    plt.title("Original Test Image")
    plt.axis('off')
    
    # Display the HOG image
    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title("HOG Features of Test Image")
    plt.axis('off')
    
    plt.show()

# Function to save features into a CSV file
def save_features_to_csv(features, filenames, file_path):
    # Create a DataFrame to save HOG features and filenames
    df = pd.DataFrame(features)
    df['filename'] = filenames  # Add filenames as an additional column
    df.to_csv(file_path, index=False)
    print(f"Features saved to {file_path}")


def main():
    folder_path = "C:/Users/A/Documents/images"  # Set the image folder path here
    images, filenames = load_and_resize_images_from_folder(folder_path)
    
    if len(images) < 1:
        print("The folder does not contain any images.")
        return
    
    # Extract HOG features for all images
    hog_features, hog_visualizations = extract_hog_features(images)
    
    # Apply PCA to reduce dimensionality of features before clustering
    reduced_features, scaler, pca = apply_pca(hog_features)

    # Apply KMeans clustering to the extracted features
    kmeans = apply_kmeans_clustering(reduced_features, n_clusters=2)

    # Save the features to CSV
    save_features_to_csv(hog_features, filenames, "C:/Users/A/Documents/images/hog_features.csv")

   
    # Load and process the test image specifically
    test_image_path = "C:/Users/A/Documents/test_image.png"  # Set the test image path here
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)  # Load test image as grayscale

    # Ensure the test image is the same size as the images used for training
    test_image_resized = cv2.resize(test_image, (128, 128))  # Resize to match input size
    
    # Compute HOG features for the test image
    test_features = hog(test_image_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    _, test_hog_image = hog(test_image_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    
    # Display the original image and HOG image
    visualize_hog_for_test_image(test_image_resized, test_hog_image)

    # Reshape the test image features to match the expected input for KMeans (similar to the training data)
    test_features_scaled = scaler.transform([test_features])  # Scale using the same scaler as for the training data
    test_reduced_features = pca.transform(test_features_scaled)  # Apply PCA transformation

    # Classify the test image using the trained KMeans model
    test_cluster = kmeans.predict(test_reduced_features)[0]  # Predict the cluster for the test image
    
    print(f"The test image belongs to Cluster {test_cluster}")

if __name__ == "__main__":
    main()
