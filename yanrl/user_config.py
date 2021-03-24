#!/usr/bin/env python
import os


# Where experiment outputs are saved by default: default = './data'
dirname = os.path.dirname
DEFAULT_DATA_DIR = os.path.join(os.path.abspath(dirname(dirname(__file__))),'data')

DEFAULT_IMG_DIR =  os.path.join(os.path.abspath(dirname(dirname(__file__))), 'img')

DEFAULT_MODEL_DIR =  os.path.join(os.path.abspath(dirname(dirname(__file__))), 'model')

DEFAULT_VIDEO_DIR =  os.path.join(os.path.abspath(dirname(dirname(__file__))), 'video')
# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True
