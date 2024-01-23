import argparse
import os
import cv2
import numpy as np
import time
import concurrent.futures
from tqdm import tqdm
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dpx import read_dpx_image_data, write_dpx, read_dpx_meta_data


def process_image(input_path, output_path, output_format='tiff', quality=None, clip=False, exportMask=None,
                  useDpx=False):
    # Carica l'immagine di input
    if useDpx:
        with open(input_path, "rb") as input_file:
            metadata = read_dpx_meta_data(input_file)
            float_img = read_dpx_image_data(input_file, metadata=metadata)
            if metadata is None or float_img is None:
                raise Exception(f"SKIPPING {input_path}: This DPX file is not valid or is unsupported")
            scale_f = 65535 / (np.power(2, metadata['depth']) - 1)
            input_img_rgb = (float_img * scale_f).astype(np.uint16)
            input_img = cv2.cvtColor(input_img_rgb, cv2.COLOR_RGB2BGR)
    else:
        input_img = cv2.imread(input_path)

    img = input_img.copy()
    max_val = np.iinfo(img.dtype).max

    # Filtro gaussiano
    img_filt = cv2.GaussianBlur(cv2.resize(img, [200, 200]), (9, 9), 1)

    # Basic lift removal
    for c in range(3):
        hist = np.cumsum(np.bincount(np.ravel(img_filt[:, :, c]), minlength=max_val + 1))
        histeq_thrs_min = max(0, np.sum(hist < 1) - 1)
        histeq_thrs_max = min(np.sum(hist < np.max(hist)) + 1, max_val)
        m = max_val / (histeq_thrs_max - histeq_thrs_min)
        lut = np.array([np.clip((m * (l - histeq_thrs_min)), 0, max_val) for l in range(max_val + 1)]).astype(img.dtype)
        img[:, :, c] = lut[img[:, :, c]]
        img_filt[:, :, c] = lut[img_filt[:, :, c]]

    img_filt_float = img_filt.astype(np.float32) / max_val

    # Calcola il canale scuro
    img_dark = np.min(img_filt_float[:, :, :3], axis=2)

    # Maschera per il canale scuro
    mask = (img_dark > np.mean(img_dark)) & (img_dark < 0.9) & (img_dark > 0.1)

    # Applica la maschera all'immagine filtrata
    masked_img_float = img_filt_float * mask[:, :, np.newaxis]
    sat = 1 - (np.max(masked_img_float[:, :, :3], axis=2) - np.min(masked_img_float[:, :, :3], axis=2))

    # Maschera di saturazione
    bins = 20
    sat_hist = np.histogram(sat[sat < 1], bins=bins)[0]
    der_cumsum_hist = np.diff(np.abs(np.diff(np.cumsum(sat_hist))))
    sat_thrs = 1
    mins_cnt = 0
    for c in range(bins - 4, 1, -1):
        if der_cumsum_hist[c + 1] > der_cumsum_hist[c] < der_cumsum_hist[c - 1]:
            mins_cnt += 1
            if mins_cnt == 1:
                sat_thrs = c * 5 / 100
                break
    mask = ((sat >= np.quantile(sat[sat < 1], sat_thrs)) & (sat < 1)).astype(img.dtype)

    mask_connected = select_connected_components(mask, sat)
    masked_img_connected = img_filt * mask_connected[:, :, np.newaxis]

    # Applica la maschera di saturazione all'immagine con maschera del canale scuro
    masked_img_not_connected = img_filt * mask[:, :, np.newaxis]

    # Calcola la media di ogni canale dei pixel non mascherati dell'immagine
    avg = [np.mean(np.ma.masked_equal(masked_img_connected[:, :, c], 0)) for c in range(3)]

    # Calcola la luminosità media delle regioni non mascherate dell'immagine originale
    avg_brightness = 0.299 * avg[0] + 0.587 * avg[1] + 0.114 * avg[2]

    img_wb = np.zeros(img.shape)
    img_norm = np.zeros(3)
    # Calcola i coefficienti del bilanciamento del bianco e correggi l'immagine originale
    for c in range(3):
        if clip:
            img_wb[:, :, c] = np.clip(img[:, :, c] * (avg_brightness / avg[c]), 0, max_val)
        else:
            img_wb[:, :, c] = img[:, :, c] * (avg_brightness / avg[c])
            img_norm[c] = (max_val / max(np.max(np.ravel(img_wb[:, :, c])), max_val))

    if not clip:
        for c in range(3):
            img_wb[:, :, c] = np.clip(img_wb[:, :, c] * np.min(img_norm), 0, max_val)

    if useDpx:
        with open(output_path, "wb+") as output_file:
            metadata['creator'] = "Luchino's fucking unsupervised white balance algorithm :) - " + metadata['creator']
            scale_f = (np.power(2, metadata['depth']) - 1) / max_val
            write_dpx(output_file, cv2.cvtColor((img_wb * scale_f).astype(np.uint16), cv2.COLOR_BGR2RGB), metadata)
    else:
        cv2.imwrite(output_path, img_wb.astype(img.dtype), params=get_save_params(output_format, quality))

    if exportMask is not None:
        cv2.imwrite(exportMask, np.hstack([masked_img_not_connected, masked_img_connected]),
                    params=get_save_params('png', None))


def select_connected_components(binary_image, sat):
    # Find connected regions in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image.astype(np.uint8))

    # Find the maximum area among all connected components
    max_area = np.max(stats[1:, cv2.CC_STAT_AREA])

    # Initialize the total selected area
    total_area = np.sum(stats[1:, cv2.CC_STAT_AREA])

    # Initialize the selected components
    selected_small_components = []
    total_small_components_area = 0
    selected_big_components = []
    total_big_components_area = 0

    # Iterate through connected components in descending order of area
    for i in np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]:
        left, top, width, height, area = stats[i + 1]  # i + 1 to account for the background component

        # Requirement 1: At least 90% of the area of the largest connected component
        if area >= 0.5 * max_area:
            selected_big_components.append(labels == i + 1)
            total_big_components_area += area
        elif total_small_components_area + total_big_components_area + area <= 0.5 * total_area:
            selected_small_components.append(labels == i + 1)
            total_small_components_area += area
        else:
            break

    # big_component_stats: column_0->variance; column_1->mean;
    if total_big_components_area > total_small_components_area:
        big_components_stats = np.empty([len(selected_big_components), 3])
        for i in range(len(selected_big_components)):
            big_components_stats[i, 0] = np.var(sat[selected_big_components[i]])
            big_components_stats[i, 1] = np.mean(sat[selected_big_components[i]])
            big_components_stats[i, 2] = i

        current_var_thrs = 1
        best_component = 0
        for i in np.argsort(big_components_stats[:, 1])[::-1]:
            if big_components_stats[i, 0] < 0.9 * current_var_thrs:
                current_var_thrs = big_components_stats[i, 0]
                best_component = i
        return selected_big_components[big_components_stats[best_component, 2].astype(np.uint8)]

    # Create the resulting image
    selected_image = np.zeros_like(binary_image)
    for component in selected_big_components:
        selected_image[component] = 1
    for component in selected_small_components:
        selected_image[component] = 1

    return selected_image


def get_save_params(output_format, quality=None):
    if output_format.lower() == 'png':
        return [cv2.IMWRITE_PNG_COMPRESSION, 5]
    elif output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if quality is None else [cv2.IMWRITE_JPEG_QUALITY, quality]
        return params
    elif output_format.lower() == 'bmp':
        return [cv2.IMWRITE_BMP_RLE, 0]
    else:
        # Default a TIFF se il formato specificato non è supportato
        return [cv2.IMWRITE_TIFF_COMPRESSION, 5]


def get_image_files(input_directory, useDpx=False):
    if useDpx:
        return [f for f in os.listdir(input_directory) if f.lower().endswith('.dpx')]
    else:
        return [f for f in os.listdir(input_directory) if
                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff'))]


def process_images_in_directory(input_directory, output_directory, output_format='tiff', quality=None,
                                clip=False, exportMask=False, useDpx=False, watch=False):
    # Assicurati che la directory di output esista
    os.makedirs(output_directory, exist_ok=True)
    if useDpx:
        output_format = 'dpx'
    if exportMask:
        os.makedirs(os.path.join(output_directory, 'masks'), exist_ok=True)

    processed_files = set()

    def on_created(event):
        nonlocal processed_files
        if not event.is_directory and event.src_path not in processed_files:
            if (event.src_path.lower().endswith('.dpx') and useDpx) or (not event.src_path.lower().endswith('.dpx') and not useDpx):
                processed_files.add(event.src_path)
                input_path = event.src_path
                output_path = os.path.join(output_directory,
                                           f"processed_{os.path.splitext(os.path.basename(input_path))[0]}.{output_format}")
                if exportMask:
                    output_path_masks = os.path.join(output_directory, 'masks',
                                                     f"mask_{os.path.splitext(os.path.basename(input_path))[0]}.png")
                else:
                    output_path_masks = None
                print(f"Processing {os.path.basename(input_path)}... ")
                time.sleep(1)
                process_image(input_path, output_path, output_format, quality, clip, output_path_masks, useDpx)
                print("done!\n\nWaiting for pictures...\n")

    event_handler = FileSystemEventHandler()
    event_handler.on_created = on_created

    if watch:
        print("Waiting for pictures...\n")
        observer = Observer()
        observer.schedule(event_handler, input_directory, recursive=False)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    else:
        image_files = get_image_files(input_directory, useDpx)
        with tqdm(image_files, desc="Processing images") as pbar:
            for image_file in pbar:
                input_path = os.path.join(input_directory, image_file)
                output_path = os.path.join(output_directory,
                                           f"processed_{os.path.splitext(image_file)[0]}.{output_format}")
                if exportMask:
                    output_path_masks = os.path.join(output_directory, 'masks',
                                                     f"mask_{os.path.splitext(image_file)[0]}.png")
                else:
                    output_path_masks = None
                process_image(input_path, output_path, output_format, quality, clip, output_path_masks, useDpx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and perform white balance on film scans.')
    parser.add_argument('input_directory', help='Input directory containing images')
    parser.add_argument('output_directory', help='Output directory for processed images')
    parser.add_argument('--format', help='Output image format (default: TIFF, accepted values: png, jpg, jpeg, bmp, '
                                         'tiff)', default='tiff')
    parser.add_argument('--quality', type=int, help='Quality setting for JPG format (default: 95)')
    parser.add_argument('--clip', action='store_true',
                        help='Clip values after equalization instead of linear rescaling ('
                             'default: False)')
    parser.add_argument('--exportMask', action='store_true',
                        help='Export white masks for debugging purposes (default: False)')
    parser.add_argument('--dpx', action='store_true',
                        help='Read and write DPX files preserving (if possible) metadata. (default: False)')
    parser.add_argument('--watch', action='store_true',
                        help='Watch for new files in the input directory and process them')

    args = parser.parse_args()

    input_directory = args.input_directory
    output_directory = args.output_directory
    output_format = args.format
    quality = args.quality
    clip = args.clip
    exportMask = args.exportMask
    useDpx = args.dpx
    watch = args.watch

    process_images_in_directory(input_directory, output_directory, output_format, quality, clip, exportMask, useDpx,
                                watch)
