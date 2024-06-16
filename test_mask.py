import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def load_mask(mask_path):
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

def overlay_mask(image, mask):
    # Create an overlay image with the mask
    overlay = image.copy()
    overlay[mask == 1] = [0, 255, 0]   # No damage - Green
    overlay[mask == 2] = [255, 255, 0] # Minor damage - Yellow
    overlay[mask == 3] = [255, 165, 0] # Major damage - Orange
    overlay[mask == 4] = [255, 0, 0]   # Destroyed - Red
    return overlay

# Load image and mask
image_path = '/home/bpeng/mnt/mnt242/scdm_data/xBD/xbd_disasters_building_polygons_neighbors/images_buffer/woolsey-fire/0006c92a-bd02-4625-bb2a-8f295bfeb85a_post_disaster.png'
mask_path = '/home/bpeng/mnt/mnt242/scdm_data/xBD/xbd_disasters_building_polygons_neighbors/masks_buffer/woolsey-fire/0006c92a-bd02-4625-bb2a-8f295bfeb85a_post_disaster.png'

image = load_image(image_path)
mask = load_mask(mask_path)

# Overlay mask on image
overlay = overlay_mask(image, mask)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title('Image with Overlayed Mask')
plt.axis('off')

# Save the image
output_path = 'test.png'
plt.savefig(output_path)

plt.show()
