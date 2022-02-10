"""
------------------------------------------------------------------------------------------------------------------
A project made by the romanian team "Alpha Robotics Team" Târgoviște for the Astro PI 2021-2022 challenge

Our project uses pycoral, reverse geocoder, the orbit module and many other libraries
in order to identify meteorological phenomena according to the types of clouds using the visible light camera,
thus transforming the raspberry into a meteorological satellite and creating a live weather forecast.

Credit to all our team members, friends and families who supported us.

Honorable mentions:

-The Astro-PI team members:
>Cristian Eduard Mihai
>Nicolau Catalin Ioan

-Out Mentor:
>Ghițeanu Ion

-Our robotics team:
>Dragomir Isabela Gabriela
>Cristian Eduard Mihai
>Nita Ionescu Constantin
>Nicolau Catalin
>Grigore Razvan Marian
>Balutoiu Bogdan Marius
------------------------------------------------------------------------------------------------------------------
"""


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

base_folder = Path(__file__).parent.resolve()

# Some csv functions
def create_csv(data_file):
    with open(data_file, 'w') as f:
        try:
            writer = csv.writer(f)
            header = ("Date/time", "Country", "City", "Weather")
            writer.writerow(header)
        except:
            logger.error("Couldn't create a csv file")

def add_csv_data(data_file, data):
    with open(data_file, 'a') as f:
        try:
            writer = csv.writer(f)
            writer.writerow(data)
        except:
            logger.error("Couldn't add csv data")

def convert(angle):
    """
    Convert a `skyfield` Angle to an EXIF-appropriate
    representation (rationals)
    e.g. 98° 34' 58.7 to "98/1,34/1,587/10"
    """
    try:
        sign, degrees, minutes, seconds = angle.signed_dms()
        exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
        return sign < 0, exif_angle
    except:
        logger.error("Couldn't convert skyfiled angle to EXIF")

def capture(camera, image):
    """Use `camera` to capture an `image` file with lat/long EXIF data."""
    try:
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
    except:
        logger.error("Couldn't capture a photo")

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

print("Hello from Romania !")

logger.info(f"I run in {base_folder}")

interpreter = make_interpreter(f"{model_file}")
interpreter.allocate_tensors()

camera = PiCamera()

# Create a datetime variable for the current time
now_time = datetime.now()

while (now_time < project_start_time + timedelta(minutes=175)):

    # Take a picture and save it in the raw img dir
    timestamp = str(now_time)
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
        try:
            weather = labels.get(c.id, c.id)
            logger.info(f'{timestamp}.jpg: {weather} {c.score:.5f}')

            coordinates = ISS.coordinates()
            coordinate_pair = (
                coordinates.latitude.degrees,
                coordinates.longitude.degrees)
            
            location = reverse_geocoder.search(coordinate_pair)
            ("Date/time", "Country", "City", "Weather")
            # What is this ^

            row = (timestamp, location[0]['cc'], location[0]['name'], weather)
            add_csv_data(data_file, row)
        except:
            logger.error("Error in the for loop")
    
    sleep(30)

    # Update the current time
    try:
        now_time = datetime.now()
    except:
        logger.error("Couldn't update the time")