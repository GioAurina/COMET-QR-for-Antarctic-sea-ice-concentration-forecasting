"""
Useful routines for post-processing of raw data. 
The data is assumed to be a list of size N_years of ndarrays of shape (N_days, ny, nx), 
where nx and ny are image sizes along x- and y- axes.
"""

import numpy as np
from scipy.signal import fftconvolve
from datetime import datetime, timedelta
from tqdm.autonotebook import tqdm, trange


def thin_data(step_thin = 1, *args):
    """
    Spatially thin any lists/ndarrays in args by the same factor step_thin.
    Spatial dimensions are assumed to be the last two dimensions of each array.
    
    step_thin : int
        step to thin data

    Example usage (thin data):
    out_data, out_y_mean_month, out_y_mean_week, out_x, out_y, out_mask_ice, out_mask_land = \
    thin_data(2, DATA, Y_mean_month, Y_mean_week, x, y, mask_ice, mask_land)
    """
    # perform thinning across all lists in args
    if step_thin==1:
        return args
    
    elif step_thin>1:

        out = []
        for arg in args:
            if type(arg) is list:
                out_arg = []
                for a in arg:
                    out_arg.append(a[:, ::step_thin, ::step_thin])
                out.append(out_arg)
                print(out_arg[0].shape)
            else:
                if arg.ndim == 3:
                    arg = arg[:, ::step_thin, ::step_thin]
                elif arg.ndim == 2:
                    arg = arg[::step_thin, ::step_thin]
                elif arg.ndim == 1:
                    arg = arg[::step_thin]
                else:
                    print('Unknown data shape')
                    return None
                out.append(arg)
               
        return out

def del_leap(*args, leap_year_0 = 3):
    """
    Delete leap years from any lists in args.
    
    Example usage (thin and de-leap data):
    out_data, out_y_mean_month, out_y_mean_week, out_x, out_y, out_mask_ice, out_mask_land = \
    thin_data(2, DATA, Y_mean_month, Y_mean_week, x, y, mask_ice, mask_land)
    out_data1 = del_leap(out_data)
    """
    leap_day = 59
    n_years = len(args[0])-1
    leap_years = [q for q in range(leap_year_0, n_years, 4)]
    
    # perform thinning across all lists in args
    out = []
    for arg in args:
        if type(arg) is list:
            out_arg = []

            for year in trange(len(arg)):
                if year in leap_years: 
                    y0 = np.delete(arg[year], leap_day, axis=0)
                    out_arg.append(y0)
                else:
                    out_arg.append(arg[year])
    
       
            out.append(out_arg)
        else:
            print('must be list')
            return None
    
    return out_arg

def day_to_date(index, reference_date = datetime(1989, 1, 1)):
    """
    day since Jan 1, 1989 (day 0) to date
    """
    target_date = reference_date + timedelta(days=index)
    return target_date.year, target_date.month, target_date.day

def date_to_day(year, month, day, reference_date = datetime(1989, 1, 1)):
    """
    date to days since Jan 1, 1989 = day 0
    """
    target_date = datetime(year, month, day)
    delta = target_date - reference_date
    return delta.days

def year_day_to_date(year, day):
    date = datetime(year, 1, 1) + timedelta(days=day )
    return date.year, date.month, date.day

def date_to_year_day(year, month, day):
    date = datetime(year, month, day)
    start_of_year = datetime(year, 1, 1)
    day_of_year = (date - start_of_year).days
    return date.year, day_of_year

def year_day_to_day(year, day, reference_date = datetime(1989, 1, 1)):
    """
    Concert (year, day) to days since January 1, 1989
    """
    year, month, day = year_day_to_date(year, day)
    target_date = datetime(year, month, day)
    delta = target_date - reference_date
    return delta.days+1

def day_to_year_day(day, reference_date = datetime(1989, 1, 1)):
    """
    Concert day since January 1, 1989 to year, day

    Days in year counted from one
    """

    year, month, day = day_to_date(day, reference_date = reference_date)

    return date_to_year_day(year, month, day)


def get_days_before(data, year_0, day_0, T):
    """
    Get T days before [year_0, day_0], not inclusive of day_0. 
    If don't have this many days, return all that have

    

    Parameters:
        data (list): List of N elements, where each element has shape (N_days, nx, ny). 
                    Note: N_days can be different for different list elements
        year_0 (int): Index of the year (0-based) in the data.
        day_0 (int): Index of the day (0-based) in the specified year.
        T (int): Length of the window for retrieving previous days' data.

    Returns:
        previous_days (numpy.ndarray): Array of shape (t, nx, ny) containing previous days' data, where t<=T.
        Note last element is the day before year_0, day_0
    """
    
    # within year_0, inclusive of year_0
    get_days_before_ = lambda data, year_0, day_0, T: data[year_0][max(day_0-T+1, 0):day_0+1]

    out = get_days_before_(data, year_0, day_0-1, T) 
    
    T -= day_0
    year_0 -= 1
    while T>=0 and year_0>=0:

        day_0 = data[year_0].shape[0]-1
        out1 = get_days_before_(data, year_0, day_0, T) 

        out = np.concatenate((out1, out), axis = 0)
        T -= day_0+1
        year_0 -= 1

    return out


def get_days_after(data, year_0, day_0, T):
    """
    Get T days after [year_0, day_0], inclusive of day_0. 
    If don't have this many days, return all that have

    Same params and return as get_days_before, but for after, inclusive of day0
    """

    assert day_0 <= data[year_0].shape[0]-1

    # within year_0, inclusive of year_0 and possibly last day
    get_days_after_ = lambda data, year_0, day_0, T: \
        data[year_0][day_0: min(day_0+T, data[year_0].shape[0])]

    out = get_days_after_(data, year_0, day_0, T)

    # days left in this year, after day_0, inclusive of day_0
    N_days_left = data[year_0].shape[0]-day_0

    T -= N_days_left
    year_0 += 1
    while T>0 and year_0 <= len(data)-1:
        out1 = get_days_after_(data, year_0, 0, T) 

        out = np.concatenate((out, out1), axis = 0)

        T -= data[year_0].shape[0]
        year_0 += 1

    return out


def window_mean(days_array, window, t = None):
    """
    Compute window-mean of days_array over a given number of days.
    The resulting configurations should be aligned in time with the end of time array

    Parameters: 
        days_array (ndarray of shape (N_days, ny, nx)): daily snapshots
        window (int): time window over which to take the mean
        t: optional array of times

    Returns: 
        windowed mean array of shape (N_days-window+1, ny, nx) of window-means
        [because window-1 first elements cannot be averaged]

        If times array given, returns a truncated times array, so that window is before current time

    Note:
        If needed to get K window-meaned configurations, call for K+window-1 snapshots, 
            extending window-1 into past
    """

    ny = days_array.shape[1]
    nx = days_array.shape[2]
    out = fftconvolve(days_array, np.ones((window, ny, nx))/window, mode = 'valid', axes = 0)
    out[out<0] = 0.

    if t is not None:
        t = t[window-1:]
        out = (out, t)

    return out


def get_test_set(DATA, year, day, window, T_test):
    """
    Perform window-averaging on days after day, year. 
    
    """

    true_after_ = get_days_after(DATA, year, day, T_test)
    true_win = get_days_before(DATA, year, day, window-1)

    true_win_after = np.concatenate((true_win, true_after_), axis = 0)
    true_after = window_mean(true_win_after, window = window, t = None)

    return true_after
