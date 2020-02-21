"""Functions return path to respective files"""
import os


def app_path():
    """Returns app root"""

    os.chdir("..")
    path = os.getcwd()
    os.chdir("notebooks")

    return path


def raw_data_path(filename):
    return os.path.join(app_path(), 'data', 'raw', filename)


def processed_data_path(filename):
    return os.path.join(app_path(), 'data', 'processed', filename)
