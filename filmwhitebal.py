from argparse import ArgumentParser
from os import listdir, makedirs, path
from time import sleep

import cv2
import numba as nb
import numpy as np
from tqdm import tqdm
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from dpx import read_dpx_image_data, write_dpx, read_dpx_meta_data


# The 1D luts to the final image is applied in parallel, which led to a reduction in processing time of >50% for larger
# images
@nb.njit(parallel=True)
def apply_lut_parallel_single_channel(img, lut):
    height, width = img.shape
    for i in nb.prange(height):
        for j in range(width):
            img[i, j] = lut[img[i, j]]
    return img


# White balance with computed coefficients is performed in parallel using one thread per channel,
# which led to a reduction in processing time of >50% for larger images
@nb.njit(parallel=True)
def white_balance_parallel(img, clip, avg_brightness, avg, max_val):
    img_wb = np.zeros_like(img, dtype=np.float32)
    img_norm = np.zeros(3, dtype=np.float32)
    for c in nb.prange(3):
        if clip:
            img_wb[:, :, c] = np.clip(img[:, :, c] * (avg_brightness / avg[c]), 0, max_val)
        else:
            img_wb[:, :, c] = img[:, :, c] * (avg_brightness / avg[c])
            img_norm[c] = (max_val / np.maximum(np.max(img_wb[:, :, c]), max_val))

    # If --clip option is not used, all the pixels values are scaled based on the absolute maximum value to prevent
    # clipping for pixels having values above dtype.max
    if not clip:
        for c in nb.prange(3):
            img_wb[:, :, c] = np.clip(img_wb[:, :, c] * np.min(img_norm), 0, max_val)

    return img_wb


def process_image(input_path, output_path, output_format='tiff', quality=None, clip=False, exportMask=None,
                  useDpx=False):
    # Load input image
    if useDpx:
        # If --dpx option is used, the binary file is read and converted to uint16 np array, with value scaled
        # between 0 and 65535 regardless of the actual bit depth.
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
    # The blurred and scaled to 200x200 image is used for analysis to save time.
    analysis_img_int = cv2.GaussianBlur(cv2.resize(img, [200, 200]), (9, 9), 1)

    # Basic lift removal: a 1D lut per channel is computed in order to remove lift.
    # img might contain values greater than dtype.max, thus need to be rescaled later.
    for c in range(3):
        hist = np.cumsum(np.bincount(analysis_img_int[:, :, c].ravel(), minlength=max_val + 1))
        histeq_thrs_min = max(0, np.sum(hist < 1) - 1)
        histeq_thrs_max = min(np.sum(hist < np.max(hist)) + 1, max_val)
        m = max_val / (histeq_thrs_max - histeq_thrs_min)
        lut = np.clip((m * (np.arange(max_val + 1) - histeq_thrs_min)), 0, max_val).astype(img.dtype)
        analysis_img_int[:, :, c] = lut[analysis_img_int[:, :, c]]
        # The LUTs to the full size image are applied in parallel
        img[:, :, c] = apply_lut_parallel_single_channel(img[:, :, c], lut)

    # The analysis image is converted to float32 and scaled between 0 and 1
    analysis_img_float = (analysis_img_int / max_val).astype(np.float32)

    # Dark channel is computed
    dark_channel_img = np.min(analysis_img_float[:, :, :3], axis=2)

    # The first mask is computed based on the dark channel of the analysis image.
    # The mask identifies the pixels having dark channel value greater than the average dark channel value for the whole
    # image. Pixels with values lower than 0.1 or greater than 0.9 are excluded to avoid using too dark or clipped
    # RGB values.
    mask = (dark_channel_img > np.mean(dark_channel_img)) & (dark_channel_img < 0.9) & (dark_channel_img > 0.1)

    # The analysis image is masked with the dark channel mask.
    masked_analysis_img_float = analysis_img_float * mask[:, :, np.newaxis]

    # The saturation map is computed. This is simply the inverse of the saturation, so 1 corresponds to min saturation
    # and 0 to max saturation.
    sat_map = 1 - (np.max(masked_analysis_img_float[:, :, :3], axis=2) - np.min(masked_analysis_img_float[:, :, :3],
                                                                                axis=2))

    # Exclude the most saturated regions, setting a threshold of 0.25 or less based on the 75th percentile of the
    # saturation values
    sat_min_thrs = min(np.quantile(sat_map, 0.75), 0.25)
    
    # Now we want to look for the least saturated regions, using as a threshold the level of minimum saturation
    # bringing the greates amount of information. This is accomplished by computing the absolute value of the second
    # derivative of the saturation cumulative histogram and using as threshold the first local minimum encountered.
    # 1) The cumulative histogram of the saturation is computed using only 20 bins (to account for noise)
    sat_hist = np.histogram(sat_map[(sat_min_thrs < sat_map) * (sat_map < 1)], bins=20)[0]
    # 2) We compute der_cumsum_hist as the absolute value of the second derivative of the cumulative histogram
    der_cumsum_hist = np.diff(np.abs(np.diff(np.cumsum(sat_hist))))
    sat_quantile = 1
    # 3) Starting from the values of least saturation (from the right of the histogram), we move toward the values of
    #    higher saturation and look for the first local minimum.
    #    From this step we take the sat_thrs = argmin(der_cumsum_hist).
    for c in range(16, 1, -1):
        if der_cumsum_hist[c + 1] > der_cumsum_hist[c] < der_cumsum_hist[c - 1]:
                sat_quantile = sat_min_thrs + (c * (1 - sat_min_thrs) * 5 / 100)
                break

    # 4) The new mask is computed by taking the dark channel mask and keeping only the pixels having value
    #    greater than or equal to the sat_thrs quantile of the saturation map.
    mask = ((sat_map >= np.quantile(sat_map[sat_map < 1], sat_quantile)) & (sat_map < 1)).astype(img.dtype)
    # mask = ((sat >= sat_thrs) & (sat < 1)).astype(img.dtype)

    # Compute the lightness map needed by selected_connected_components to filter out regions having great MSE in
    # luminosity, preferring more uniform ones.
    light_map = 0.299 * analysis_img_float[:, :, 2] + 0.587 * analysis_img_float[:, :, 1] + 0.114 * \
                analysis_img_float[:, :, 0]

    # We filter out the computed mask based on the connected components.
    # Aim of this step is to keep only the biggest, least saturated and most uniform regions.
    mask_connected = select_connected_components(mask, sat_map, light_map)

    # The analysis image is masked with the obtained mask.
    masked_img_connected = (analysis_img_int * mask_connected[:, :, np.newaxis]).astype(img.dtype)

    # The analysis image is masked with the mask without the connected regions-based filtering, this is used only
    # for debugging purposes with --exportMask option enabled.
    masked_img_not_connected = (analysis_img_int * mask[:, :, np.newaxis]).astype(img.dtype)

    # We compute avg[] as each channel's average value from the masked analysis image.
    avg = [np.mean(np.ma.masked_equal(masked_img_connected[:, :, c], 0)) for c in range(3)]

    # From avg[] we computed avg_brightness as the average brightness of the unmasked regions.
    avg_brightness = 0.299 * avg[2] + 0.587 * avg[1] + 0.114 * avg[0]

    # We compute the white balance coefficients as coeff[c] = avg_brightness/avg[c].
    # Using the computed coefficients all the pixels in the actual input img is scaled.
    img_wb = white_balance_parallel(img, clip, avg_brightness, avg, max_val)

    # If --dpx option is used, the file is written using write_dpx function.
    if useDpx:
        with open(output_path, "wb+") as output_file:
            metadata['creator'] = "Luchino's fucking unsupervised white balance algorithm :) - " + metadata['creator']
            scale_f = (np.power(2, metadata['depth']) - 1) / max_val
            write_dpx(output_file, cv2.cvtColor((img_wb * scale_f).astype(np.uint16), cv2.COLOR_BGR2RGB), metadata)
    else:
        cv2.imwrite(output_path, img_wb.astype(img.dtype), params=get_save_params(output_format, quality))

    # If --exportMask option is used, a "masks" folder is created in the output directory and both normal and
    # connected-regions-filtered masks are written in PNG files.
    if exportMask is not None:
        cv2.imwrite(exportMask, np.hstack([analysis_img_int, masked_img_not_connected, masked_img_connected]),
                    params=get_save_params('png', None))


def select_connected_components(binary_image, sat_map, light_map):
    # Find connected regions in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image.astype(np.uint8))

    # Initialize the total selected area
    total_area = np.sum(stats[1:, cv2.CC_STAT_AREA])

    # Component score calculation: for each component, we compute a score (between 0 and 1) which is a result of the
    # product of two indicators: squareness and fullness. Squareness amounts to 1 for a component whose bounding box is
    # square and decreases as the bounding box becomes more of a rectangle; fullness amounts to 1 when the area of the
    # component equals that of the bounding box and decreases as the area of the component lowers.
    # Saturation stats calculation: for each component, the average saturation and the variance in saturation of the
    # component is computed.
    # All the stats are saved in the component_scores_array array.
    components_scores = []
    for i in np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]:
        left, top, width, height, area = stats[i + 1]  # i + 1 to account for the background component
        var_light = np.power(1 - np.var(light_map[labels == i + 1]), 2)
        fullness = max(1, (area / (np.power(min(width, height) / 2, 2) * np.pi)))
        # fullness = np.power(area / (width * height), 3)
        squareness = min(width, height) / max(width, height)
        avg_sat = np.mean(sat_map[labels == i + 1])
        var_sat = np.var(sat_map[labels == i + 1])
        components_scores.append(
            [i + 1, var_light * fullness * squareness, area, avg_sat, var_sat, var_light])
    components_scores_array = np.array(components_scores)

    # We filter out the smaller components based on the 90% percentile of the area of the regions. A minimum threshold
    # of 64 is set.
    quant = max(64, (np.quantile(components_scores_array[1:, 2], 0.9)))

    selected_components_scores = []
    selected_area = 0
    # We sort the components in descending order based on the previously computed score
    for i in np.argsort(components_scores_array[0:, 1])[::-1]:
        # We discard the smaller components
        if components_scores_array[i, 2] >= quant:
            # We select the components in order until we reach at least 10% of the total area of the components
            selected_components_scores.append(components_scores_array[i, :])
            selected_area = selected_area + components_scores_array[i, 2]
            if selected_area > 0.25 * total_area:
                break

    # In case we couldn't find any component satisfying any of the conditions, we return the whole binary image,
    # thus ignoring the connected-regions-based filtering.
    if selected_area == 0:
        return binary_image

    selected_image = np.zeros_like(binary_image)
    selected_components_scores_array = np.array(selected_components_scores)

    avg_var = np.var(selected_components_scores_array[0:, 4])
    last_sat = 0
    # We iterate through the selected components from the previous step in descending order based on the average
    # saturation. Remember that the saturation map is computed as the inverse of the saturation, so we are actually
    # iterating based on saturation in ascending order.
    for i in np.argsort(selected_components_scores_array[0:, 3])[::-1]:
        # The first least saturated region is selected
        selected_image[labels == selected_components_scores_array[i, 0]] = 1
        # We keep selecting regions having an average saturation value of at least 90% that of the previous (to include
        # similar regions) and with a saturation variance which is less than the average saturation variance among
        # all regions.
        if selected_components_scores_array[i, 3] < 0.9 * last_sat and selected_components_scores_array[i, 4] < avg_var:
            break
        last_sat = selected_components_scores_array[i, 3]

    # We return the mask as a binary image
    return selected_image


# Get command line parameters.
def get_save_params(output_format, quality=None):
    if output_format.lower() == 'png':
        return [cv2.IMWRITE_PNG_COMPRESSION, 5]
    elif output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if quality is None else [cv2.IMWRITE_JPEG_QUALITY, quality]
        return params
    elif output_format.lower() == 'bmp':
        return [cv2.IMWRITE_BMP_RLE, 0]
    else:
        # Default to TIFF if given format is not supported
        return [cv2.IMWRITE_TIFF_COMPRESSION, 5]


# Scan the input directory for files.
def get_image_files(input_directory, useDpx=False, inputExtension=None):
    if useDpx:
        return [f for f in listdir(input_directory) if f.lower().endswith('.dpx')]
    else:
        return [f for f in listdir(input_directory) if
                (inputExtension is None and f.lower().endswith((
                    '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')))
                or (inputExtension is not None and f.lower().endswith(inputExtension))]


# Process each image in the input directory.
def process_images_in_directory(input_path, output_directory, output_format='tiff', quality=None,
                                clip=False, exportMask=False, useDpx=False, watch=False, outputName=None,
                                inputExtension=None):
    if not path.exists(input_path):
        print("Input file or directory does not exist.")
        return -1
    if path.isfile(input_path) and watch:
        print("Option --watch can be used on an input directory, input file given.")
        return -1

    makedirs(output_directory, exist_ok=True)
    if useDpx:
        output_format = 'dpx'
    if exportMask:
        makedirs(path.join(output_directory, 'masks'), exist_ok=True)

    processed_files = set()

    if outputName is None:
        prepend_str = ''
    else:
        prepend_str = f"{outputName}_"

    def on_created(event):
        nonlocal processed_files
        if not event.is_directory and event.src_path not in processed_files:
            if (event.src_path.lower().endswith('.dpx') and useDpx) or (
                    not event.src_path.lower().endswith('.dpx') and not useDpx):
                processed_files.add(event.src_path)
                input_path = event.src_path
                output_path = path.join(output_directory,
                                        f"{prepend_str}{path.splitext(path.basename(input_path))[0]}.{output_format}")
                if exportMask:
                    output_path_masks = path.join(output_directory, 'masks',
                                                  f"mask_{path.splitext(path.basename(input_path))[0]}.png")
                else:
                    output_path_masks = None
                print(f"Processing {path.basename(input_path)}... ")
                sleep(1)
                process_image(input_path, output_path, output_format, quality, clip, output_path_masks, useDpx)
                print("done!\n\nWaiting for pictures...\n")

    event_handler = FileSystemEventHandler()
    event_handler.on_created = on_created

    if watch:
        print("Waiting for pictures...\n")
        observer = Observer()
        observer.schedule(event_handler, input_path, recursive=False)
        observer.start()
        try:
            while True:
                sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    else:
        if path.isfile(input_path):
            image_name = path.basename(input_path)
            image_files = [image_name]
            input_path = path.dirname(input_path)
            if image_name.lower().endswith('.dpx') and not useDpx:
                print("DPX file format detected, forcing --dpx option.")
                useDpx = True
                output_format = 'dpx'
        else:
            image_files = get_image_files(input_path, useDpx, inputExtension)
        with tqdm(image_files, desc="Processing images") as pbar:
            for image_file in pbar:
                input_image_path = path.join(input_path, image_file)
                output_path = path.join(output_directory,
                                        f"{prepend_str}{path.splitext(image_file)[0]}.{output_format}")
                if exportMask:
                    output_path_masks = path.join(output_directory, 'masks',
                                                  f"mask_{path.splitext(image_file)[0]}.png")
                else:
                    output_path_masks = None
                process_image(input_image_path, output_path, output_format, quality, clip, output_path_masks, useDpx)


if __name__ == "__main__":
    parser = ArgumentParser(description='Luchino\'s fucking unsupervised white balance algorithm for films :)\n'
                                        'Process images and perform white balance on film scans.')
    parser.add_argument('input_path', help='Input file or directory containing images.')
    parser.add_argument('output_directory', help='Output directory for processed images.')
    parser.add_argument('--outputName', type=str, help='Prepend string followed by underscore to the output filename.')
    parser.add_argument('--dpx', action='store_true',
                        help='Read and write DPX files preserving (if possible) metadata. (default: False)')
    parser.add_argument('--format', help='Output image format. Ignored if --dpx option is used. (default: TIFF, '
                                         'accepted values: png, jpg, jpeg, bmp, tiff)', default='tiff')
    parser.add_argument('--inputExtension', type=str, help='Load only images with given extension in input directory. '
                                                           'Ignored if --dpx option is used.')
    parser.add_argument('--quality', type=int, help='Quality setting for JPG format. (default: 95)')
    parser.add_argument('--clip', action='store_true',
                        help='Clip values after equalization instead of linear rescaling. ('
                             'default: False)')
    parser.add_argument('--exportMask', action='store_true',
                        help='Export white masks for debugging purposes. (default: False)')
    parser.add_argument('--watch', action='store_true',
                        help='Watch for new files in the input directory and process them.')

    args = parser.parse_args()

    input_path = args.input_path
    output_directory = args.output_directory
    output_format = args.format
    quality = args.quality
    clip = args.clip
    exportMask = args.exportMask
    useDpx = args.dpx
    watch = args.watch
    outputName = args.outputName
    inputExtension = args.inputExtension

    process_images_in_directory(input_path, output_directory, output_format, quality, clip, exportMask, useDpx,
                                watch, outputName, inputExtension)
