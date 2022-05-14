import gspread
from oauth2client.service_account import ServiceAccountCredentials

CREDENTIALS_FILE = "CREDS"
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]


def authorize_and_return_worksheet() -> gspread.worksheet.Worksheet:
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, SCOPE)
    client = gspread.authorize(creds)
    return client.open("mlpractice").worksheet("grades")


def update_ws_stats(worksheet: gspread.worksheet.Worksheet, name, stats):
    names = worksheet.col_values(1)[2:]  # [2:], because first two cells are 'Name' and ''
    len_col_name = len(names) + 2
    if name not in names:
        worksheet.update_cell(len_col_name + 1, 1, name)  # +1, because numeration in table starts from 1
        names.append(name)
        len_col_name += 1

    ind = names.index(name) + 3  # +2 for 'Name', '' and +1, because numeration in table starts from 1
    cell_lists = [worksheet.range(ind, 2, ind, 6), worksheet.range(ind, 7, ind, 8)]
    for cell_list, task in zip(cell_lists, stats.keys()):
        for cell, problem in zip(cell_list, stats[task].keys()):
            cell.value = stats[task][problem]
        worksheet.update_cells(cell_list)
