import json
import os.path
import datetime

file_path = './logs/ab_testing_logs.json'

datetime_format = "%d.%m.%Y, %H:%M:%S"


def write_logs_to_file(request, response):
    if not os.path.isfile(file_path):
        create_file()
    with open(file_path, 'r+') as file:
        file_data = json.load(file)
        data_entry = {
            'request': request,
            'response': response,
            'timestamp': datetime.datetime.now().strftime(datetime_format)
        }
        file_data['logs'].append(data_entry)
        file.seek(0)
        json.dump(file_data, file, indent=4)


def create_file():
    curr_time = datetime.datetime.now().strftime(datetime_format)

    data = {
        'meta_data': {
            'file_created_at': curr_time,
        },
        'logs': []
    }

    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
