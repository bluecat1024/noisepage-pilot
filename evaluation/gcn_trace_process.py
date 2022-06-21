import csv
import json
import sys

TOTAL_COLUMN_COUNT = 26
STATEMENT_COLUMN = 13
CLIENT_COLUMN = 23

POSTGRES_EPOCH_JDATE = 2451545
UNIX_EPOCH_JDATE = 2440588
SECS_PER_DAY = 86400

# Get start_time and the EXPLAIN ANALYZE plan tree for sample query in the middle.
def sample_workload(workload_csv, sample_count = 100000):
    with open(workload_csv, 'r') as fr:
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

        fw = open(workload_csv + 'processed.txt', 'w')
        print(len(json_list))
        assert len(json_list) > sample_count
        for json_str in json_list[(len(json_list) - sample_count) // 2 : (len(json_list) - sample_count) // 2 + sample_count]:
            fw.write(json_str + '\n')
    
    return

if __name__ == '__main__':
    assert len(sys.argv) > 1
    sample_workload(sys.argv[1])