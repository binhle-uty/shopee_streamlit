import re

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