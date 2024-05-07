from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

from to_minimap import To_Minimap

f = To_Minimap()
#path_image is a path to an image
path_image = 'KpSFR/dataset/WorldCup_2014_2018/TS-WorldCup/TS-WorldCup/Dataset/80_95/right/2014_Match_Highlights1_clip_00015-1/IMG_001.jpg'
image = Image.open(path_image)

fig, ax = plt.subplots(1, 2, figsize=(8, 3))  # 1 row, 2 columns

# Plot a minimap image with detected players
ax[0].imshow(cv.resize(np.array(image), (1050, 680), interpolation=cv.INTER_CUBIC))
ax[0].axis('off')  # Turn off axis

ax[1].imshow(f.to_minimap_with_detection(image))
ax[1].axis('off')

# Display the plot
plt.tight_layout()
plt.show()
