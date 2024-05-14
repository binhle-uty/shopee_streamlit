import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.misc import derivative

def convert_google_sheet_url(url: str = None):
    """Function to convert the url to get dataframe from pandas
    """
    # Regular expression to match and capture the necessary part of the URL
    pattern = r'https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)(/edit#gid=(\d+)|/edit.*)?'
    # Replace function to construct the new URL for CSV export
    # If gid is present in the URL, it includes it in the export URL, otherwise, it's omitted
    replacement = lambda m: f'https://docs.google.com/spreadsheets/d/{m.group(1)}/export?' + (f'gid={m.group(3)}&' if m.group(3) else '') + 'format=csv'
    # Replace using regex
    new_url = re.sub(pattern, replacement, url)
    return new_url



def convert_money_to_number(money:str) -> float:
    result = 0
    if money:
        [num, dec] = money.rsplit(',')
        result += int(num.replace('.', ''))
        result += (int(dec) / 100)
    return result


def calculate_trend(list_value: list):
    y = list_value
    x = np.arange(1, len(y) + 1, 1)
    # Simple interpolation of x and y
    f = interp1d(x, y, fill_value="extrapolate")
    x_fake = np.arange(1.1, 30, 0.1)

    # derivative of y with respect to x
    df_dx = derivative(f, x_fake, dx=1e-6)

    average = np.average(df_dx)

    print("Max trend measure is:")
    print(np.max(df_dx))
    print("min trend measure is:")
    print(np.min(df_dx))
    print("Overall trend measure:")
    print(((np.max(df_dx))-np.min(df_dx)-average)/((np.max(df_dx))-np.min(df_dx)))

    max_trend = np.max(df_dx)
    min_trend = np.min(df_dx)
    overall_trend = ((np.max(df_dx))-np.min(df_dx)-average)/((np.max(df_dx))-np.min(df_dx))

    extermum_list_y = []
    extermum_list_x = []

    for i in range(0,df_dx.shape[0]):
        if df_dx[i] < 0.001 and df_dx[i] > -0.001:
            extermum_list_x.append(x_fake[i])
            extermum_list_y.append(df_dx[i])

    return average, max_trend, min_trend, overall_trend, extermum_list_x, extermum_list_y