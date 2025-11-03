import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import matplotlib.pyplot as plt
import os

input_raster_path ="F:/Project/Data/NAIP Water/water_clipped.tif"
output_dir = "processed_images"
os.makedirs(output_dir, exist_ok=True)

# Phase 1: Data Extraction & Preparation

print(f"Loading data from: {input_raster_path}")

try:
    with rasterio.open(input_raster_path) as src:
        profile = src.profile
        bounds = src.bounds
        crs = src.crs

        red_band = src.read(1)
        green_band = src.read(2)
        blue_band = src.read(3)
        no_data_val = src.nodata

        print(f"Original image band shape: {red_band.shape} (per band)")
        print(f"NoData value: {no_data_val}")

        minx_roi, miny_roi, maxx_roi, maxy_roi = bounds.left + 100, bounds.bottom + 100, bounds.right - 100, bounds.top - 100
        roi_polygon = box(minx_roi, miny_roi, maxx_roi, maxy_roi)
        geoms = [roi_polygon.__geo_interface__]

        print(f"Clipping image to ROI: {geoms}")

        clipped_red_data, transform = mask(src, geoms, crop=True, indexes=1)
        clipped_green_data, _ = mask(src, geoms, crop=True, indexes=2)
        clipped_blue_data, _ = mask(src, geoms, crop=True, indexes=3)

        print("clipped_red_data raw shape after mask:", clipped_red_data.shape)
        print("clipped_green_data raw shape after mask:", clipped_green_data.shape)
        print("clipped_blue_data raw shape after mask:", clipped_blue_data.shape)

        if clipped_red_data.ndim == 3 and clipped_red_data.shape[0] == 1:
            clipped_red_2d = clipped_red_data[0]
            clipped_green_2d = clipped_green_data[0]
            clipped_blue_2d = clipped_blue_data[0]
        else: # Assume it's already (H, W)
            clipped_red_2d = clipped_red_data
            clipped_green_2d = clipped_green_data
            clipped_blue_2d = clipped_blue_data

        print("clipped_red 2D shape:", clipped_red_2d.shape)
        print("clipped_green 2D shape:", clipped_green_2d.shape)
        print("clipped_blue 2D shape:", clipped_blue_2d.shape)
        
        rows, cols = clipped_red_2d.shape 

        profile.update(
            transform=transform,
            height=rows,
            width=cols,
            count=3  
        )

        # Prepare data for FFT and filtering 
        def fill_nodata_for_fft(img_array, nodata_value):
            if nodata_value is not None:
                return np.ma.masked_equal(img_array, nodata_value).filled(0)
            return img_array

        input_for_fft_r = fill_nodata_for_fft(clipped_red_2d, no_data_val)
        input_for_fft_g = fill_nodata_for_fft(clipped_green_2d, no_data_val)
        input_for_fft_b = fill_nodata_for_fft(clipped_blue_2d, no_data_val)

        print(f"Clipped image shape for FFT: {input_for_fft_r.shape} (per band)")

        rgb_clipped_display = np.stack([input_for_fft_r, input_for_fft_g, input_for_fft_b], axis=-1)
        print("RGB stacked shape for display:", rgb_clipped_display.shape)

except rasterio.errors.RasterioIOError as e:
    print(f"Error loading raster: {e}. Please check the path and file integrity.")
    exit() 
except Exception as e:
    print(f"An unexpected error occurred during data loading or clipping: {e}")
    exit()

# Phase 2: Implementing Fourier Transform Filtering 

print("Starting Fourier Transform filtering...")

def fft_filter_image(image_band, filter_mask):
    f = np.fft.fft2(image_band)
    fshift = np.fft.fftshift(f)
    fshift_filtered = fshift * filter_mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back), fshift

rows, cols = input_for_fft_r.shape
crow, ccol = rows // 2, cols // 2

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: 
        center = (int(w/2), int(h/2))
    if radius is None: 
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return dist_from_center

# Ideal Low-Pass Filter
ilpf_radius = 30 
ilpf_mask = create_circular_mask(rows, cols, radius=ilpf_radius)
ilpf_mask = ilpf_mask < ilpf_radius

#Ideal High-Pass Filter (IHPF) 
ihpf_radius = .2 
ihpf_mask = create_circular_mask(rows, cols, radius=ihpf_radius)
ihpf_mask = ihpf_mask > ihpf_radius 

# Gaussian Low-Pass Filter 
glpf_sigma = 30 
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
gaussian_mask = np.exp(-((X - ccol)**2 + (Y - crow)**2) / (2 * glpf_sigma**2))

# ILPF
img_ilpf_r, fshift_r_ilpf = fft_filter_image(input_for_fft_r, ilpf_mask)
img_ilpf_g, fshift_g_ilpf = fft_filter_image(input_for_fft_g, ilpf_mask)
img_ilpf_b, fshift_b_ilpf = fft_filter_image(input_for_fft_b, ilpf_mask)

# IHPF
img_ihpf_r, fshift_r_ihpf = fft_filter_image(input_for_fft_r, ihpf_mask)
img_ihpf_g, fshift_g_ihpf = fft_filter_image(input_for_fft_g, ihpf_mask)
img_ihpf_b, fshift_b_ihpf = fft_filter_image(input_for_fft_b, ihpf_mask)

# GLPF
img_glpf_r, fshift_r_glpf = fft_filter_image(input_for_fft_r, gaussian_mask)
img_glpf_g, fshift_g_glpf = fft_filter_image(input_for_fft_g, gaussian_mask)
img_glpf_b, fshift_b_glpf = fft_filter_image(input_for_fft_b, gaussian_mask)

print("Filtering complete.")

# Phase 3:Visualization 
plt.figure(figsize=(15, 16))

rgb_ilpf = np.stack([img_ilpf_r, img_ilpf_g, img_ilpf_b], axis=-1)
rgb_ihpf = np.stack([img_ihpf_r, img_ihpf_g, img_ihpf_b], axis=-1)
rgb_glpf = np.stack([img_glpf_r, img_glpf_g, img_glpf_b], axis=-1)

def normalize_image_for_display(img_array):
    if img_array.ndim == 3: # RGB image
        img_min = img_array.min()
        img_max = img_array.max()
        scaled = (img_array - img_min) / (img_max - img_min + 1e-8) 
    else: # Grayscale image
        img_min = img_array.min()
        img_max = img_array.max()
        scaled = (img_array - img_min) / (img_max - img_min + 1e-8)
    return (scaled * 255).astype(np.uint8)

plt.subplot(221), plt.imshow(normalize_image_for_display(rgb_clipped_display))
plt.title('Original (Clipped) RGB Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(normalize_image_for_display(rgb_ilpf))
plt.title(f'Ideal Low-Pass (Radius={ilpf_radius})'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(normalize_image_for_display(rgb_ihpf))
plt.title(f'Ideal High-Pass (Radius={ihpf_radius})'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(normalize_image_for_display(rgb_glpf))
plt.title(f'Gaussian Low-Pass (Sigma={glpf_sigma})'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

print("Exporting filtered images for QGIS...")

def save_geotiff_rgb(data_stack, output_path, profile_template, dtype='float32'):
    out_profile = profile_template.copy()
    out_profile.update(
        dtype=dtype,
        count=3,
        nodata=None # Or set a specific nodata value if desired
    )
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(data_stack[:, :, 0].astype(dtype), 1) # Red
        dst.write(data_stack[:, :, 1].astype(dtype), 2) # Green
        dst.write(data_stack[:, :, 2].astype(dtype), 3) # Blue
    print(f"Saved: {output_path}")

save_geotiff_rgb(rgb_ilpf, os.path.join(output_dir, "filtered_ilpf_rgb.tif"), profile)
save_geotiff_rgb(rgb_ihpf, os.path.join(output_dir, "filtered_ihpf_rgb.tif"), profile)
save_geotiff_rgb(rgb_glpf, os.path.join(output_dir, "filtered_glpf_rgb.tif"), profile)

print("\nAll filtered color images have been exported to the 'processed_images' directory.")