import time
import datetime
import gymnasium as gym
import numpy as np

def extract_dict_values(d):
    """
    Extracts all values from a nested dictionary with any depth.
    """
    values_list = []

    for key, value in d.items():
        if isinstance(value, dict):
            values_list.extend(extract_dict_values(value))
        else:
            values_list.append(value)

    return values_list


def ymd_hms():
    """
    Returns the current time in year, month, day, hours, minutes, and seconds.
    """
    return convert_gmt_to_pacific(time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()))


def convert_gmt_to_pacific(gmt_time):
    """
    Converts a GMT time string to Pacific time.
    """
    gmt_time = datetime.datetime.strptime(gmt_time, "%Y-%m-%d_%H-%M-%S")
    pacific_time = gmt_time - datetime.timedelta(hours=7)
    return pacific_time.strftime("%Y-%m-%d_%H-%M-%S")

def seconds_to_hms(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%02d:%02d:%02d" % (hours, minutes, seconds)


def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Mask function that masks out the actions that are not available in the current state.
    :param env: (gym.Env)
    :return: (np.ndarray)
    """
    return env.action_mask()