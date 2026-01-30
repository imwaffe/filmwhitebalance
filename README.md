# Luchino's fucking unattended white balance algorithm for films

Please do not distribute. It's just an alpha, unlike me :(

The main function is in [filmwhitebal.py](./filmwhitebal.py).
The windows executable is downloadable from the release.

### Dependencies
If using pip, use the following to get all the required packages.
~~~
py -m pip install argparse opencv.python numpy tqdm watchdog numba==0.59.0rc1
~~~

### Usage
This is a command line program, following is the syntax (if using the executable).
~~~
usage: filmwhitebal [-h] [--outputName OUTPUTNAME] [--dpx] [--format FORMAT] [--inputExtension INPUTEXTENSION]
                       [--quality QUALITY] [--clip] [--exportMask] [--watch]
                       input_path output_directory

Luchino's fucking unsupervised white balance algorithm for films :) Process images and perform white balance on film
scans.

positional arguments:
  input_path            Input file or directory containing images.
  output_directory      Output directory for processed images.

options:
  -h, --help            show this help message and exit
  --outputName OUTPUTNAME
                        Prepend string followed by underscore to the output filename.
  --dpx                 Read and write DPX files preserving (if possible) metadata. (default: False)
  --format FORMAT       Output image format. Ignored if --dpx option is used. (default: TIFF, accepted values: png,
                        jpg, jpeg, bmp, tiff)
  --inputExtension INPUTEXTENSION
                        Load only images with given extension in input directory. Ignored if --dpx option is used.
  --quality QUALITY     Quality setting for JPG format. (default: 95)
  --clip                Clip values after equalization instead of linear rescaling. (default: False)
  --exportMask          Export white masks for debugging purposes. (default: False)
  --watch               Watch for new files in the input directory and process them.
~~~

### DPX supported
Currently only few DPX file formats are supported:
- 16 bits, packed into 32 bits words
- 10 bits, filled to 32 bit words, with 2 bits left padding
- Non-encoded (Run-Length encoding is not supported)
- Non-compressed
- RGB color descriptor (all the RGBA and YCbCr descriptors are not supported)
