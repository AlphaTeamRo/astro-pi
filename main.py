from pathlib import Path
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

from picamera import PiCamera
from datetime import datetime, timedelta
import re
import os
from time import sleep

import reverse_geocoder
from orbit import ISS

import csv
from logzero import logger, logfile

# Start the 3h timer
project_start_time = datetime.now()

# Create a datetime variable for the current time
now_time = datetime.now()

base_folder = Path(__file__).parent.resolve()

# Some csv functions
def create_csv(data_file):
    with open(data_file, 'w') as f:
        writer = csv.writer(f)
        header = ("Date/time", "Country", "City", "Weather")
        writer.writerow(header)

def add_csv_data(data_file, data):
    with open(data_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def convert(angle):
    """
    Convert a `skyfield` Angle to an EXIF-appropriate
    representation (rationals)
    e.g. 98° 34' 58.7 to "98/1,34/1,587/10"
    """
    sign, degrees, minutes, seconds = angle.signed_dms()
    exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
    return sign < 0, exif_angle

def capture(camera, image):
    """Use `camera` to capture an `image` file with lat/long EXIF data."""
    point = ISS.coordinates()

    # Convert the latitude and longitude to EXIF-appropriate representations
    south, exif_latitude = convert(point.latitude)
    west, exif_longitude = convert(point.longitude)

    # Set the EXIF tags specifying the current location
    camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
    camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
    camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"

    # Capture the image
    camera.capture(image)

# the TFLite converted to be used with edgetpu
model_file = f'{base_folder}/models/model_edgetpu.tflite'

# The path to labels.txt that was downloaded with your model
label_file = f'{base_folder}/models/labels.txt'

# The path to the raw photos dir
img_dir = f'{base_folder}/raw'

# The main csv file, where the meteorological data will be recorded
data_file = f'{base_folder}/data.csv'

create_csv(data_file)

# The events log file
logfile(f'{base_folder}/events.log')


interpreter = make_interpreter(f"{model_file}")
interpreter.allocate_tensors()

camera = PiCamera()

while (now_time < project_start_time + timedelta(hours=3)):

    # Take a picture and save it in the raw img dir
    timestamp = str((datetime.now()))
    timestamp = timestamp[0:19]
    timestamp = re.sub(r'[:]', '-', re.sub(r'[ ]', '_', timestamp))
    #camera.capture(f"{base_folder}/raw/{timestamp}.jpg")
    capture(camera, f"{base_folder}/raw/{timestamp}.jpg")

    image_file = f"{base_folder}/raw/{timestamp}.jpg"

    size = common.input_size(interpreter)
    image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

    common.set_input(interpreter, image)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)

    labels = read_label_file(label_file)
    for c in classes:
        weather = labels.get(c.id, c.id)
        logger.info(f'{timestamp}.jpg: {weather} {c.score:.5f}')

        if weather == 'night':
            os.remove(image_file)
            logger.info(f'Removed night time photo {timestamp}.jpg in order to save space')
        else:
            coordinates = ISS.coordinates()
            coordinate_pair = (
                coordinates.latitude.degrees,
                coordinates.longitude.degrees)
            location = reverse_geocoder.search(coordinate_pair)
            ("Date/time", "Country", "City", "Weather")
            row = (timestamp, location[0]['cc'], location[0]['name'], weather)
            add_csv_data(data_file, row)
    
    # Update the current time
    now_time = datetime.now()

    
    sleep(30)