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
def sample_workload(folder_path, csv_path, sample_count = 100000, batch_size = 100):
    assert sample_count % batch_size == 0
    with open(csv_path, 'r') as fr:
        reader = csv.reader(fr)
        json_list = []
        for row in reader:
            if len(row) != TOTAL_COLUMN_COUNT or row[CLIENT_COLUMN] != 'client backend':
                continue

            if not row[STATEMENT_COLUMN].startswith('duration:'):
                continue

            raw_text = row[STATEMENT_COLUMN]
            raw_json = json.loads(raw_text[raw_text.find('plan:') + len('plan:'):])
            new_json = {
                'start_time' : raw_json['start_time'] * 1e-6 + (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE) * SECS_PER_DAY,
                'plan': [[[{'Plan': raw_json['Plan']}]]],
            }

            json_list.append(json.dumps(new_json))

        try:
            shutil.rmtree(folder_path + '_processed/')
            os.mkdir(folder_path + '_processed/')
        except:
            pass

        
        print(len(json_list))
        assert len(json_list) > sample_count

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
    for path in os.listdir(all_collected_path):
        if not path.startswith(benchmark) or not os.path.isdir(all_collected_path + path) or path.endswith('processed'):
            continue
        sample_workload(all_collected_path + path, all_collected_path + path + '/workload.csv')