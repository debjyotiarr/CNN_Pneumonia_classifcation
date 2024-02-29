import matplotlib.pyplot as plt

# Function to visualize images from the generator
def visualize_images(generator, num_images=5):
    # Iterate over the generator to get batches of images and labels
    for i, (images, labels) in enumerate(generator):
        # Print the batch index
        print("Batch", i)
        # Iterate over the images in the batch
        for j in range(len(images)):
            # Visualize the image
            plt.imshow(images[j])
            plt.title(f"Label: {labels[j]}")
            plt.axis('off')
            plt.show()
            # Break after visualizing num_images images
            if j == num_images - 1:
                break
        # Break after visualizing num_images batches
        if i == num_images - 1:
            break

