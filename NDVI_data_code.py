import numpy as np
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import os

input_tiff = "F:\\Project\\NDVI\\drive-download-20251102T125123Z-1-001\\Mumbai_NDVI.tif" 
output_dir = "F:\\Project\\result_raw\\NDVI"
os.makedirs(output_dir, exist_ok=True)

try:
    with rasterio.open(input_tiff) as src:
        ndvi = src.read(1)                    
        profile = src.profile                 
        no_data = src.nodata
        print("NDVI loaded successfully")
except Exception as e:
    print("Error reading file:", e)
    raise SystemExit

if no_data is not None:
    ndvi = np.ma.masked_equal(ndvi, no_data).filled(0)

def apply_fft_filter(image, mask):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
    return np.abs(img_back)

rows, cols = ndvi.shape
crow, ccol = rows // 2, cols // 2

X, Y = np.ogrid[:rows, :cols]
distance = np.sqrt((X - crow)**2 + (Y - ccol)**2)

# Low-Pass Filter 
radius = 100           
low_pass_mask = distance < radius

# High-Pass Filter 
high_pass_mask = distance > radius

ndvi_low_pass = apply_fft_filter(ndvi, low_pass_mask)
ndvi_high_pass = apply_fft_filter(ndvi, high_pass_mask)

def normalize(img):
    img_min, img_max = np.min(img), np.max(img)
    return (255 * (img - img_min) / (img_max - img_min)).astype(np.uint8)

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.title("Original NDVI")
plt.imshow(normalize(ndvi), cmap='RdYlGn')
plt.colorbar(label='NDVI')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Low-Pass (Smoothed NDVI)")
plt.imshow(normalize(ndvi_low_pass), cmap='RdYlGn')
plt.colorbar(label='NDVI')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("High-Pass (Enhanced Edges)")
plt.imshow(normalize(ndvi_high_pass), cmap='gray')
plt.colorbar(label='NDVI Edge Strength')
plt.axis('off')

plt.tight_layout()
plt.show()
