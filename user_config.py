#!/usr/bin/env python
import os


# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),'data')

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True
