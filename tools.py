import rasterio
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from rasterio.transform import TransformMethodsMixin
from rasterio.warp import calculate_default_transform, reproject, Resampling
import earthaccess
import sys
from scipy.ndimage import binary_dilation
from pyproj import Transformer
import random
from collections import Counter
from cupyx.scipy.ndimage import convolve
import cupy as cp
import leafmap
from samgeo import SamGeo, tms_to_geotiff, get_basemaps

def view_rgb(img):
    min_val = np.min([img[5], img[3], img[1]])
    max_val = np.max([img[5], img[3], img[1]])
    
    def normalize_band(band):
        return (band - min_val) / (max_val - min_val)
    
    red = normalize_band(img[5])
    green = normalize_band(img[3])
    blue = normalize_band(img[1])
    
    rgb_1 = np.stack([red, green, blue], axis=-1)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_1)
    plt.show()

def get_bounding_box(geotiff_path):
    with rasterio.open(geotiff_path) as dataset:

        transform = dataset.transform
        width = dataset.width
        height = dataset.height
        
        crs = dataset.crs

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        top_left = (transform.c, transform.f)
        bottom_right = (transform * (width, height))
        
        min_x, max_y = top_left
        max_x, min_y = bottom_right
        min_lon, max_lat = transformer.transform(min_x, max_y)
        max_lon, min_lat = transformer.transform(max_x, min_y)
        
        # Return bounding box in lat/lon
        return (min_lon, min_lat, max_lon, max_lat)
    
def download_dem (tif_path, dem_path=r'C:\Users\attic\planet_mask\dem'): 
    bounding_box = get_bounding_box(tif_path)
    if bounding_box is None:
        print("Invalid tile")
        return
    earthaccess.login(persist=True)
    dem_results = earthaccess.search_data(
        short_name="ASTGTM",
        bounding_box=bounding_box)
    for result in dem_results:
        earthaccess.download(result, local_path=dem_path)

def reproject_dem_to_img(hls_path, dem_path=r'C:\Users\attic\HLS_Kelp\imagery\Socal_DEM.tiff'):
    with rasterio.open(hls_path) as dst:
        img = dst.read()
        dem = rasterio.open(dem_path)
        # plt.figure()
        # dem_plot = dem.read()
        # plt.imshow(dem_plot[0])
        # plt.show()
        if dem.crs != dst.crs:
            reprojected_dem = np.zeros((img.shape[1], img.shape[2]), dtype=img.dtype)
            reproject(
                source=dem.read(),
                destination=reprojected_dem,
                src_transform=dem.transform,
                src_crs=dem.crs,
                dst_transform=dst.transform,
                dst_crs=dst.crs,
                resampling=Resampling.bilinear)
            if reprojected_dem.any():
                return reprojected_dem
            else:
                return None
            
def compile_dem(dem_path, hls_path):
    files = os.listdir(dem_path)
    dem_files = [file for file in files if '_dem' in file]
    dem = None
    for file in dem_files:
        if(dem is None):
            dem = (reproject_dem_to_img(hls_path=hls_path, dem_path=os.path.join(dem_path,file)))
        else:
            dem = np.where(dem == 0, reproject_dem_to_img(hls_path=hls_path, dem_path=os.path.join(dem_path,file)), dem)
    dem = dem.astype(np.float64)
    dem = np.where(np.isnan(dem), 0, dem)

    return dem 

def generate_land_mask(reprojected_dem, land_dilation=7, show_image=False, as_numpy=True):
    if reprojected_dem.any():
        if(land_dilation > 0):
            struct = np.ones((land_dilation, land_dilation))
            reprojected_dem_gpu = np.asarray(reprojected_dem)
            land_mask = binary_dilation(reprojected_dem_gpu > 0, structure=struct)
        elif(land_dilation < 0):
            struct = np.ones((-land_dilation, -land_dilation))
            reprojected_dem_gpu = np.asarray(reprojected_dem)
            land_mask = ~binary_dilation(~(reprojected_dem_gpu > 0), structure=struct)
        else:
            land_mask = reprojected_dem_gpu > 0
        # if as_numpy:
        #     land_mask = np.asnumpy(land_mask)
        if show_image:
            plt.figure(figsize=(6, 6))
            if as_numpy:
                plt.imshow(land_mask, cmap='gray')
            else:
                plt.imshow((land_mask))
            plt.show()
        return land_mask
    else:
        print("Something failed, you better go check...")
        sys.exit()

def create_coastal_strip(land_mask):
    struct = np.ones((2, 2))
    seed = np.where(~land_mask, binary_dilation(land_mask, structure=struct),False)

    struct = np.ones((100,100))
    landstrip = binary_dilation(seed,structure=struct)
    plt.figure()
    plt.imshow(landstrip)
    plt.show()
    
def create_land_mask(hls_path, dem_path='/mnt/c/Users/attic/HLS_Kelp/imagery/Socal_DEM.tiff', show_image=False, as_numpy=True, land_dilation=5):
    reprojected_dem = compile_dem(dem_path,hls_path)
    return generate_land_mask(reprojected_dem, show_image=show_image, land_dilation=land_dilation, as_numpy=as_numpy)

def select_ocean_endmembers(ocean_mask=None, ocean_data=None, n=30, min_pixels=2000):
    ocean_EM_n = 0
    if ocean_mask is not None:
        ocean_data = ocean_mask.reshape(ocean_mask.shape[0], -1)
        nan_columns = np.isnan(ocean_data).any(axis=0)  # Remove columns with nan 
        ocean_EM_stack =[]
        filtered_ocean = ocean_data[:, ~nan_columns]
        if len(filtered_ocean[0,:]) < min_pixels:
            print("Too few valid ocean end-members")
            return None
        i = 0
        while len(ocean_EM_stack) < n and i < 3000:
            index = random.randint(0,len(filtered_ocean[0])-1)
            ocean_EM_stack.append(filtered_ocean[:,index])
            i = i+1
        if(len(ocean_EM_stack) < 30):
            print("Invalid ocean EM selection")
            return None
        ocean_EM_stack = np.stack(ocean_EM_stack,axis=1)
        ocean_EM = np.average(ocean_EM_stack,axis=1)
    else: 
        ocean_EM = [247.14368, 295.33229714, 300.88470857, 171.69613714, 169.04841143, 140.99008, 137.85385143, 131.50875429] #default ocean endmember
    kelp_EM1 = [9.4648, 161.3627,  28.6116, 117.6279,  53.4311,  71.47, 495.9367, 523.4057]
    kelp_EM2 = [ 162.818, 371.599, 243.52566667, 367.876, 260.176, 243.66633333, 578.363, 1013.926]
    endmembers = np.array([ocean_EM,kelp_EM1,kelp_EM2]).T

    return endmembers

def unmix_ocean(kelp_mask, ocean_EM, n=3):
    height = kelp_mask.shape[1]
    width = kelp_mask.shape[2]
    kelp_data = kelp_mask.reshape(kelp_mask.shape[0], -1)
    non_zero_mask = kelp_data != 0

    #kelp_data[non_zero_mask] = (kelp_data[non_zero_mask] / np.sum(kelp_data[non_zero_mask], axis=0)) * 100
    frac1 = np.full((height, width, n), np.nan)
    frac2 = np.full((height, width, n), np.nan)
    rmse = np.full((height, width, n), np.nan)

    ocean_EM = np.array(ocean_EM)
    kelp_data = np.array(kelp_data)
    print(rmse.shape)

    print("Running MESMA")
    for k in range(n):
        B = ocean_EM[:, k].reshape(-1, 1)  # Ensure B is 2D (column vector)
        
        em_inv = np.linalg.pinv(B)  

        F = em_inv @ kelp_data

        model = (F.T @ B.T).T
        
        resids = (kelp_data - model) / 10000
        rmse[:, :, k] = np.sqrt(np.mean(resids**2, axis=0)).reshape(height, width)
        
        # Fraction of the endmember in each pixel
        frac1[:, :, k] = F[0, :].reshape(height, width)

    minVals = np.nanmin(rmse, axis=2)
    PageIdx = np.nanargmin(rmse, axis=2)
    rows, cols = np.meshgrid(np.arange(rmse.shape[0]), np.arange(rmse.shape[1]), indexing='ij')
    Zindex = np.ravel_multi_index((rows, cols, PageIdx), dims=rmse.shape)
    Mes2 = frac1.ravel()[Zindex]
    Mes2 = Mes2
    Mes2 = -0.229 * Mes2**2 + 1.449 * Mes2 - 0.018 #Landsat mesma corrections 
    Mes2 = np.clip(Mes2, 0, None)  # Ensure no negative values
    Mes2 = np.round(Mes2 * 100).astype(np.int16)
    return Mes2, minVals

def run_mesma(kelp_mask, ocean_EM, n=3):
    height = kelp_mask.shape[1]
    width = kelp_mask.shape[2]
    kelp_data = kelp_mask.reshape(kelp_mask.shape[0], -1)
    
    pixels = kelp_data.shape[1]
    sand_EM = [1098.51958333, 1561.91916667, 1890.74166667, 2290.19958333, 2571.75708333, 2685.44541667, 2736.12583333, 3051.03]
    sand_EM = (sand_EM / np.sum(sand_EM))*100
    non_zero_mask = ocean_EM != 0
    ocean_EM[non_zero_mask] = (ocean_EM[non_zero_mask] / np.sum(ocean_EM, axis=0)) * 100
    frac1 = np.full((pixels, n), np.nan)
    frac2 = np.full((pixels, n), np.nan)
    rmse = np.full((pixels, n), np.nan)

    ocean_EM = np.array(ocean_EM)
    kelp_data = np.array(kelp_data)
    print(rmse.shape)
    for k in range(n):
        B = np.column_stack((ocean_EM[:, k], sand_EM))
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        IS = Vt.T / S
        em_inv = IS @ U.T
        F = em_inv @ kelp_data
        #print(F)
        model = (F.T @ B.T).T
        resids = (kelp_data - model) / 10000
        rmse[:, k] = np.sqrt(np.mean(resids**2,axis=0 ))
        frac1[:, k] = F[0,:]
        frac2[:, k] = F[1,:]

    rmse = np.asarray(rmse)
    minVals = np.nanmin(rmse, axis=1)
    # print(minVals)
    PageIdx = np.nanargmin(rmse, axis=1)
    rows = np.arange(rmse.shape[0])

    PageIdx = np.expand_dims(PageIdx, axis=0)
    Zindex = np.ravel_multi_index((rows,  PageIdx), dims=rmse.shape)
    Mes2 = frac2.ravel()[Zindex]
    Mes1 = frac1.ravel()[Zindex]
    sand_number = Mes2.reshape(height,width)
    ocean_number = Mes1.reshape(height,width)
    min_vals_2D = minVals.reshape(height,width)
    return ocean_number,sand_number, min_vals_2D

def convert_to_8bit_rgb(input_path, output_path):

    with rasterio.open(input_path) as src:
        img_data = src.read([6,4,2])
        no_data = src.nodata
        profile = src.profile
        if no_data == 65535.0:
            img_data = np.where(img_data > 65500.0, 0, img_data)

    img_data_norm = np.zeros_like(img_data, dtype=np.uint8)
    for i in range(img_data.shape[0]): 

        img_data_norm[i] = np.interp(img_data[i], (img_data[i].min(), img_data[i].max()), (0, 255)).astype(np.uint8)

        profile.update(
        dtype=rasterio.uint8,  # 8-bit data type
        count=3,  # 3 bands for RGB
        nodata=None,  # Set nodata value to 0 for 8-bit
    )
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(img_data_norm)

    print(f"Image saved:{output_path}")

def noise_filter(mask, count=5, neighborhood=3):
    mask = cp.array(mask)
    kernel = cp.ones((neighborhood, neighborhood), dtype=cp.int32)

    mask_conv = convolve(mask.astype(cp.int32), kernel, mode='constant', cval=0.0)
    #print(f'kelp moving average finished: {time.time()-time_val}')

    mask = cp.where(((~mask) | (mask_conv <= count)), 0, 1) 
    return cp.asnumpy(mask)