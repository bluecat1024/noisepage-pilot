import csv
import json
import sys
import os
import shutil


TOTAL_COLUMN_COUNT = 26
STATEMENT_COLUMN = 13
CLIENT_COLUMN = 23

POSTGRES_EPOCH_JDATE = 2451545
UNIX_EPOCH_JDATE = 2440588
SECS_PER_DAY = 86400

# Get start_time and the EXPLAIN ANALYZE plan tree for sample query in the middle.
def sample_workload(folder_path, csv_path,
    sample_count = 7500000, start_count=100000, serial=False):
    if serial:
        sample_workload(folder_path + '10', csv_path,
            6000000, start_count=100000)
        sample_workload(folder_path + '50', csv_path,
            6000000, start_count=10000000)
        sample_workload(folder_path + '100', csv_path,
            6000000, start_count=19000000)

        return

    batch_size = 1000
    assert sample_count % batch_size == 0

    curr_query_cnt = 0
    recorded_query_cnt = 0
    with open(csv_path, 'r') as fr:
        reader = csv.reader(fr)
        json_list = []
        for row in reader:
            if len(row) != TOTAL_COLUMN_COUNT or row[CLIENT_COLUMN] != 'client backend':
                continue

            if not row[STATEMENT_COLUMN].startswith('{\n  "Query Text"'):
                continue
            
            curr_query_cnt += 1
            if curr_query_cnt < start_count:
                continue

            raw_json = json.loads(row[STATEMENT_COLUMN])
            new_json = {
                'start_time' : raw_json['start_time'] * 1e-6 + (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE) * SECS_PER_DAY,
                'plan': [[[{'Plan': raw_json['Plan'], 'Triggers': raw_json['Triggers']}]]],
                'query_text' : raw_json['Query Text'],
                'query_id' : raw_json['query_id'],
                'elapsed_us': raw_json['elapsed_us'],
            }

            if (len(json_list) < sample_count):
                json_list.append(json.dumps(new_json))

            recorded_query_cnt += 1
            # print(len(json_list))

            if len(json_list) >= sample_count:
                break

        try:
            shutil.rmtree(folder_path + '_processed/')
        except:
            pass
        os.mkdir(folder_path + '_processed/')

        
        print(len(json_list))
        print(recorded_query_cnt)
        assert len(json_list) == sample_count

        sampled_list = json_list[(len(json_list) - sample_count) // 2 : (len(json_list) - sample_count) // 2 + sample_count]
        for idx in range(sample_count // batch_size):
            fw = open(folder_path + f"_processed/sample-plan-{idx}", 'w')
            for json_str in sampled_list[idx * batch_size: (idx + 1) * batch_size]:
                fw.write(json_str + '\n')
            fw.close()
    
    return

if __name__ == '__main__':
    assert len(sys.argv) > 2
    all_collected_path = sys.argv[1]
    benchmark = sys.argv[2]
    try:
        is_serial = int(sys.argv[3])
    except:
        is_serial = 0

    for path in os.listdir(all_collected_path):
        if not path.startswith(benchmark) or not os.path.isdir(all_collected_path + path) or path.endswith('processed')\
            or (is_serial == 0 and path.endswith('serial'))\
            or (is_serial != 0 and not path.endswith('serial')):
            continue
        sample_workload(all_collected_path + path, all_collected_path + path + '/workload.csv',\
            serial=(is_serial == 1))