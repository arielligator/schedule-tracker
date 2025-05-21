# IMPORTS
import csv, io
import pandas as pd
import dropbox
import re
import requests
from pprint import pprint
from datetime import datetime
import streamlit as st

# GET CSV FROM DROPBOX
CLIENT_ID     = st.secrets["dropbox"]["client_id"]
CLIENT_SECRET = st.secrets["dropbox"]["client_secret"]
REFRESH_TOKEN = st.secrets["dropbox"]["refresh_token"]
TOKEN_URL = 'https://api.dropboxapi.com/oauth2/token'

def refresh_dropbox_token():
    """
    Refresh the Dropbox access token using the provided refresh token.
    """
    data = {
        'grant_type': 'refresh_token',
        'refresh_token': REFRESH_TOKEN
    }

    # Use HTTP Basic Authentication with your client credentials.
    response = requests.post(TOKEN_URL, data=data, auth=(CLIENT_ID, CLIENT_SECRET))
    
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data.get('access_token')
        #print("New access token:", access_token)
        return access_token
    else:
        print("Error refreshing DropBox token:")
        print(response.status_code, response.text)
        return None

# Retrieve a fresh Dropbox access token and create the Dropbox client
dropbox_access_token = refresh_dropbox_token()
if dropbox_access_token is None:
    raise Exception("Failed to obtain Dropbox access token.")
dbx = dropbox.Dropbox(dropbox_access_token)

# Specify the path to the CSV file in Dropbox
dropbox_file_path = '/Daily Schedule/Daily Schedule.csv'

# download the file into memory
metadata, res = dbx.files_download(dropbox_file_path)


#============================================================================================================================================
# LOAD CSV INTO DICTIONARY
s = res.content.decode('utf-8') # turn bytes into str
f = io.StringIO(s) # wrap it in a file-like object

reader = csv.DictReader(f) # read from that
rows = [
    {key.strip(): value.rstrip()
     for key, value in row.items()}
    for row in reader]

# Fix location header and add new columns
old_A1 = reader.fieldnames[0]
new_location = 'Location'

# Remove trailing space from 'Name' key
old_name, new_name = 'Name ', 'Name'

for row in rows:
    row[new_location] = row.pop(old_A1) # replace A1 with 'Location'
    # row[new_name] = row.pop(old_name) #remove trailing space
    row['Start Time'] = ''
    row['End Time'] = ''
    row['Work Day Range'] = ''
    row['Work Days'] = ''
    row['Work Dates'] = ''




#============================================================================================================================================
# HELPER FUNCTIONS - TEXT EXTRACTION

def extract_parentheses(text):
    # extract text within parentheses from a given string
    if not text or not isinstance(text, str):
        return ""
    
    match = re.search(r'\((.*?)\)', text)
    return match.group(1) if match else ""

def split_on_hyphen(text: str) -> tuple[str, str]:
    before, sep, after = text.partition('-')
    return before.strip(), after.strip()


# HANDLE WEIRD INDIVIDUAL CASES

# Handle KB
KB = next(row for row in rows if st.secrets["name"]["name1"] in row['Name'])
kb_suffix = extract_parentheses(KB['Shift'])
KB['Name'] = f"{KB['Name']} ({kb_suffix})"
KB['Shift'] = '8:30am - 5:30pm'
#KB is row 49

# Handle MM
MM = next(row for row in rows if st.secrets["name"]["name2"] in row['Name'])
MM['Shift'] = '8:00am - 5:30pm'
MM['Work Day Range'] = 'Monday - Thursday'

MM2 = {
    'Team': 'MGE',
    'Name': st.secrets["name"]["name2"],
    'Shift': '8:00am - 12:00pm',
    'Lunch': '',
    'Location': 'NYC',
    'Start Time': '8:00',
    'End Time': '12:00',
    'Work Day Range': 'Friday',
    'Work Days': ['Friday'],
    'Work Dates': ''
}

rows.append(MM2)

# Handle rotating schedules
for row in rows:
if any("Rotating" in r.get("Shift", "") for r in rows):
    rotating_emp = next(row for row in rows if 'Rotating' in row['Shift'])
    rotating_suffix = extract_parentheses(row['Shift'])
    rotating_prefix = row['Shift'] - rotating_suffix
    row['Name'] = row['Name'] + rotating_suffix
    row['Shift'] = rotating_prefix
    row['Work Day Range'] = 'Monday - Saturday'

    # CP = next(row for row in rows if st.secrets["name"]["name3"] in row['Name'])
    # CP['Name'] = st.secrets["name"]["name3"] + ' (Rotating Days/Week)'
    # CP['Shift'] = '6:00am - 6:30pm'
    # CP['Work Day Range'] = 'Monday - Saturday'

    # kroud = next(row for row in rows if st.secrets["name"]["name4"] in row['Name'])
    # kroud['Name'] = st.secrets["name"]["name4"] + ' (Rotating Days/Week)'
    # kroud['Shift'] = '7:00am - 3:30pm'
    # kroud['Work Day Range'] = 'Monday - Saturday'

#============================================================================================================================================
# EXTRACT START/END TIMES FROM SHIFT

for row in rows:
    row['Start Time'], row['End Time'] = split_on_hyphen(row['Shift'])

# EXTRACT WORK DAYS

def parse_workdays(day_range_str):
    """
    Given a string representing a range of days (e.g., "Saturday &  Sunday" or "Saturday - Sunday"),
    returns a list of days covered by that range.
    
    If the input isn't exactly in that format, returns None.
    """
    # remove leading and trailing whitespace
    day_range_str = day_range_str.strip()

    if not any(delim in day_range_str for delim in ['-', '&', '/']):
        return [day_range_str]

    # Regular expression to capture two day names separated by "-" or "&"
    # This pattern expects letters for the day names and optional whitespace on either side.
    regex = r'^\s*([A-Za-z]+)\s*(?:-|&)\s*([A-Za-z]+)\s*$'
    
    match = re.match(regex, day_range_str)
    if not match:
        return None  # Format doesn't match
    
    start_day_input, end_day_input = match.group(1), match.group(2)
    
    # Define ordered days (adjust if needed for your locale)
    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    # Create a lookup for case-insensitive matching
    day_lookup = {day.lower(): day for day in days}
    
    start_day = day_lookup.get(start_day_input.lower())
    end_day   = day_lookup.get(end_day_input.lower())
    
    if start_day is None or end_day is None:
        return None  # Invalid day names
    
    start_index = days.index(start_day)
    end_index   = days.index(end_day)
    
    if start_index <= end_index:
        return days[start_index:end_index + 1]
    else:
        return days[start_index:] + days[:end_index + 1]

for row in rows:
    if not row['Work Day Range']:
        row['Work Day Range'] = extract_parentheses(row['Shift'])
    if row['Work Day Range']:
        row['Work Days'] = parse_workdays(row['Work Day Range'])

for row in rows:
    if not row['Work Days']:
        row['Work Days'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# REMOVE PARENTHETICALS FROM END TIME COLUMN
def remove_parentheses(text: str) -> str:
    """
    Remove any parenthetical and leading whitespace from `text`.
    E.g. "Saturday (Holiday)" → "Saturday"
    """
    # \s*         → any whitespace before the “(”
    # \( [^)]* \) → a literal “(… )” with no “)” inside
    cleaned = re.sub(r'\s*\([^)]*\)', '', text)
    return cleaned.strip()

for row in rows:
    row['End Time'] = remove_parentheses(row['End Time'])

# CONVERT TIMES TO 24-HR SCALE
def to_military(time_str: str) -> str:
    ts = time_str.strip().lower().replace(" ", "")
    # 1) handle missing-colon cases like "800am", "130pm", "9pm"
    m = re.match(r'^(\d{1,4})(am|pm)$', ts)
    if m:
        num, period = m.groups()
        # split numeric part into hours/minutes
        if len(num) <= 2:
            hour = int(num)
            minute = 0
        else:
            hour = int(num[:-2])
            minute = int(num[-2:])
        # adjust for 12-hour edge cases
        if period == 'am' and hour == 12:
            hour = 0
        elif period == 'pm' and hour != 12:
            hour += 12
        return f"{hour:02d}:{minute:02d}"

    # 2) fallback to standard colon-based parsing
    for fmt in ("%I:%M%p", "%I:%M %p"):
        try:
            dt = datetime.strptime(time_str.strip(), fmt)
            return dt.strftime("%H:%M")
        except ValueError:
            pass

    raise ValueError(f"Unrecognized time format: {time_str!r}")

for row in rows:
    row['Start Time'] = to_military(row['Start Time'])
    row['End Time'] = to_military(row['End Time'])

# pprint(rows)

# CONVERT TO DATAFRAME

df = pd.DataFrame(rows)
omit = ['Shift','Work Day Range']
show_cols = [c for c in df.columns if c not in omit]
df = df[show_cols]
df_order = ['Name', 'Team', 'Location', 'Work Days','Start Time', 'End Time', 'Lunch']
df = df[df_order]
df['Work Days'] = df['Work Days'].str.join(', ')

# print(df.to_string())
