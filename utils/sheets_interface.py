"""
author: aa
"""

import os

from apiclient.discovery import build
from httplib2 import Http
from oauth2client import client, file, tools
from redis import WatchError

CREDENTIALS_PATH = os.path.join(os.getcwd()[:os.getcwd().index('WormholeLearning')],
                                'WormholeLearning',
                                "resources/credentials/")

# Setup the Sheets API
SCOPES = 'https://www.googleapis.com/auth/spreadsheets'

# Spreadsheet ID
SPREADSHEET_ID = '1ayYY_QUQqsv-FsqGZ7hMlHzPxh9ivbg3UgmHdKrtP9Y'  # TFRecords Builder result


class GoogleSheetsInterface:
    def __init__(self, credentials='google_sheets_credentials.json'):
        self._credentials = credentials
        store = file.Storage(os.path.join(CREDENTIALS_PATH, 'google_sheets_auth.json'))
        creds = store.get()
        if not creds or creds.invalid:
            flow = client.flow_from_clientsecrets(
                    os.path.join(CREDENTIALS_PATH, self._credentials), SCOPES)
            creds = tools.run_flow(flow, store)
        self._service = build('sheets', 'v4', http=creds.authorize(Http()))
        self._get_all_sheets()

    def _get_all_sheets(self, ):
        """
        Downloads all worksheets in the specified spreadsheet ID.
        :return:
        """
        print('Initializing Google Sheets API...')

        # Get all spreadsheets
        request = self._service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID)
        response = request.execute()
        self._sheets = dict.fromkeys([sheet['properties']['title'] for sheet in response['sheets']])

        # Get all results in spreadsheets
        ranges = [('%s!A:A' % sheet) for sheet in self._sheets.keys()]
        request = self._service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID, ranges=ranges,
                                                   includeGridData=True)
        response = request.execute()
        for sheet in response['sheets']:
            self._sheets[sheet['properties']['title']] = []
            for idx, row in enumerate(sheet['data'][0]['rowData']):
                if 'effectiveValue' in row['values'][0]:
                    value = row['values'][0]['effectiveValue']['stringValue']
                    if value != '':
                        self._sheets[sheet['properties']['title']].append(
                                {'row': idx, 'network': value})
        # self.pretty_print()

    def _get_result_row(self, sheet):
        """
        Gets the next free row in the corresponding google sheet.
        :param sheet:
        :return:
        """
        # for row in self._sheets[sheet]:
        #     if network in row['network']:
        #         return row['row'] + 1
        return len(self._sheets[sheet]) + 1

    # fixme deprecated
    def upload_evaluation(self, network, testset, aps, mAP, min_obj_size=0):
        sheet = 'evaluation'
        row = self._get_result_row(sheet)
        net_range = '%s!A%i' % (sheet, row)
        val_range = '%s!E%i:N%i' % (sheet, row, row)

        # todo make this test depend on the labelset loaded!!
        if len(aps) < 7:
            values = [[testset] + [min_obj_size] + [ap for ap in aps.values()] + [mAP]]
            values[0].insert(7, '')
        else:
            values = [[testset] + [min_obj_size] + [ap for ap in aps.values()] + [mAP]]

        net_body = {'values': [[network]]}
        val_body = {'values': values}
        value_input_option = 'RAW'
        net_request = self._service.spreadsheets().values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=net_range, body=net_body,
                valueInputOption=value_input_option)
        val_request = self._service.spreadsheets().values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=val_range, body=val_body,
                valueInputOption=value_input_option)

        # Try 5 times
        for i in range(5):
            try:
                response = net_request.execute()
                response = val_request.execute()
                print('Results uploaded to Google Sheets!')
                return
            except (ConnectionError, TimeoutError, WatchError) as e:
                e += 'Error: Bad Request!'

    def upload_data(self, sheet, range_from, range_to, title, values):
        """
        More general method to upload data to the TFStatistician Google Sheet.
        By specifiying the sheet name one can upload to different destinations.
        Range from and range to should specify column letters in the corr. Google Sheet.
        :param sheet:
        :param range_from:
        :param range_to:
        :param title:
        :param values:
        :return:
        """
        # Update Google Sheet Data
        self._get_all_sheets()

        # Built Request Messages
        row = self._get_result_row(sheet)
        net_range = '%s!A%i' % (sheet, row)
        val_range = '%s!%s%i:%s%i' % (sheet, range_from, row, range_to, row)

        net_body = {'values': [[title]]}
        val_body = {'values': [values]}
        value_input_option = 'RAW'
        net_request = self._service.spreadsheets().values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=net_range, body=net_body,
                valueInputOption=value_input_option)
        val_request = self._service.spreadsheets().values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=val_range, body=val_body,
                valueInputOption=value_input_option)

        # Try to send requests 5 times
        for i in range(5):
            try:
                response = net_request.execute()
                response = val_request.execute()
                print('Results uploaded to Google Sheets!')
                return
            except (ConnectionError, TimeoutError, WatchError) as e:
                e += 'Error: Bad Request!'


if __name__ == '__main__':
    print("This is only for testing, do not run as main!")
    sheet = GoogleSheetsInterface()

    # Get Test Results
    test_mAP = 0.12

    import sys

    PROJECT_ROOT = os.getcwd()[:os.getcwd().index('objdetection')]
    sys.path.append(PROJECT_ROOT)
    from objdetection.rgb2ir.magic_constants import test_aps, test_corestats

    sheet.upload_evaluation('TEST_NN', 'TEST_DATASET', test_aps, test_mAP, test_corestats)
