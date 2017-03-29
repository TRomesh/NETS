from __future__ import print_function
import httplib2
import os

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

import dateutil.parser
from datetime import date
from operator import itemgetter
import sys

# Based on https://developers.google.com/google-apps/calendar/quickstart/python

CLIENT_SECRET_FILE = 'client_secret.json'  # Use your secret file
calendarId = 'primary'  # Use calendar 'ID' unless primary
output_dir = 'LSTM_TSU/data/inputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(CLIENT_SECRET_FILE):
    print('Not found client_secret.json. See https://developers.google.com/google-apps/calendar/quickstart/python')
    sys.exit(1)

MINUTE_NORM = 30
print_valid_events = False

try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/calendar-python-quickstart.json
SCOPES = 'https://www.googleapis.com/auth/calendar.readonly'
APPLICATION_NAME = 'Google Calendar Events Fetching & Pre-processing'


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir, 'calendar-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else:  # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials


def is_valid_title(title):
    if 'NULL' == title:
        return False
    elif 'New Event' == title:
        return False
    elif title.startswith('Call From '):  # auto-generated
        return False
    # elif title.startswith('Call To '):  # usually auto-generated or to-do
    #     return False
    elif title.startswith('Missed Call From '):  # auto-generated
        return False
    elif title.startswith('Flight to ') or title.startswith('Stay at '):  # auto-generated
        return False
    elif title.startswith('I entered http://') or title.startswith('I exited http://'):  # auto-generated
        return False
    return True


def is_valid_duration(duration):
    # most likely auto generated phone call events
    if duration.seconds % 60 > 0:
        return False
    # skip too long events: over 12 hrs
    if duration.days > 0 or duration.seconds > 3600 * 12:
        return False
    # skip length equal or less than 0 duration
    if duration.seconds // 60 <= 0 or duration.days < 0:
        return False
    return True


def dict_count(cnt_dict, cnt_key):
    num = cnt_dict.get(cnt_key)
    cnt_dict[cnt_key] = 1 if num is None else num + 1


def write_file(output_file_name, vld_events):
    with open(output_file_name, 'w', newline='', encoding='utf-8') as x_file:
        for evt_features in vld_events:
            x_file.write('\t'.join(str(e_f) for e_f in evt_features))
            x_file.write('\n')


def delete_invalid_chars_4_filename(file_name, invalid_chars):
    valid_file_name = ''

    for c in file_name:
        if c not in invalid_chars:
            valid_file_name += c
    return valid_file_name


def main():
    """Shows basic usage of the Google Calendar API.

    Creates a Google Calendar API service object and outputs a list of the next
    10 events on the user's calendar.
    """
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('calendar', 'v3', http=http)

    primary_calendar_id = None
    page_token = None
    while True:
        calendar_list = service.calendarList().list(pageToken=page_token).execute()
        for calendar_list_entry in calendar_list['items']:
            # print(calendar_list_entry['summary'])
            if calendar_list_entry.get('primary') is True:
                primary_calendar_id = calendar_list_entry['id']
        page_token = calendar_list.get('nextPageToken')
        if not page_token:
            break
    print('primary_calendar_id', primary_calendar_id)

    # now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
    print('Getting calendar events of', calendarId)
    events_result = service.events().list(
        calendarId=calendarId,
        # timeMin=now,
        # maxResults=10,
        singleEvents=True, orderBy='startTime').execute()
    events = events_result.get('items', [])

    valid_events = list()
    week_sequence_dict = {}
    filtered_events_num = {}

    if not events:
        print('No upcoming events found.')
    for event in events:
        # print(event)

        # filtering: invalid titles
        if event.get('summary') is None:
            dict_count(filtered_events_num, 'no title')
            continue
        elif not is_valid_title(event['summary']):
            dict_count(filtered_events_num, 'invalid title')
            continue

        # filtering: skip all-day events
        if event['start'].get('dateTime') is None:
            dict_count(filtered_events_num, 'all-day')
            continue

        start = event['start'].get('dateTime', event['start'].get('date'))
        start_date = dateutil.parser.parse(start)
        end_date = dateutil.parser.parse(event['end'].get('dateTime', event['end'].get('date')))
        # print(start_date, end_date)

        # filtering: phone call or too long duration
        duration = end_date - start_date
        if not is_valid_duration(duration):
            dict_count(filtered_events_num, 'invalid duration')
            continue

        start_iso_year, start_iso_week_num, _ = date(start_date.year, start_date.month, start_date.day).isocalendar()

        week_register_sequence = week_sequence_dict.get((start_iso_year, start_iso_week_num))
        if week_register_sequence is None:
            week_register_sequence = 0  # start with 0
        else:
            week_register_sequence += 1
        week_sequence_dict[(start_iso_year, start_iso_week_num)] = week_register_sequence

        # y
        start_slot = start_date.minute // MINUTE_NORM + start_date.hour * int(60 / MINUTE_NORM) + \
            start_date.weekday() * int((60 * 24) / MINUTE_NORM)

        # If you change the order of event features, check itemgetter parameters below.
        evt_features = list()
        evt_features.append(start_iso_year)
        evt_features.append(start_iso_week_num)
        evt_features.append(week_register_sequence)
        evt_features.append(duration.seconds // 60)
        evt_features.append(event['summary'])
        evt_features.append(start_slot)  # y
        valid_events.append(evt_features)

        if print_valid_events:
            print(evt_features)

    print('\n#events', len(events))
    print('#valid_events', len(valid_events))
    for fek in filtered_events_num:
        print('#filtered', fek, filtered_events_num.get(fek))

    # write a file
    invalid_chars = ['\0', '\\', '/', '*', '?', '"', '<', '>', '|']  # Unix, Windows
    output_file = output_dir + '/' + delete_invalid_chars_4_filename(primary_calendar_id.replace('@', '_at_'),
                                                                     invalid_chars) + '_events.txt'
    if os.path.exists(output_file):
        print('Overwrite existing file:', output_file)
    else:
        print('Save a file:', output_file)
    # sort by year, week and register_sequence
    sorted_valid_events = sorted(valid_events, key=itemgetter(0, 1, 2))
    write_file(output_file, sorted_valid_events)

if __name__ == '__main__':
    main()
