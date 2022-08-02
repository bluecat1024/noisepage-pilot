from plumbum import cli
import binascii
import time
import csv
import argparse
import logging
import multiprocessing as mp
import re
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from distutils import util
from enum import Enum, auto, unique

import psutil
import setproctitle
import psycopg
from psycopg.rows import dict_row
from behavior import BENCHDB_TO_TABLES
from behavior.datagen.pg_collector_utils import SettingType, _time_unit_to_ms, _parse_field, KNOBS


logger = logging.getLogger("collector")


# Name of output file/target --> (query, frequent)
PG_COLLECTOR_TARGETS = {
    "pg_stats": "SELECT EXTRACT(epoch from NOW())*1000000 as time, pg_stats.* FROM pg_stats WHERE schemaname = 'public';",
    "pg_class": "SELECT EXTRACT(epoch from NOW())*1000000 as time, * FROM pg_class t JOIN pg_namespace n ON n.oid = t.relnamespace WHERE n.nspname = 'public';",
    "pg_index": "SELECT EXTRACT(epoch from NOW())*1000000 as time, * FROM pg_index;",
    "pg_attribute": "SELECT EXTRACT(epoch from NOW())*1000000 as time, * FROM pg_attribute;",
}


def pg_collector(output_rows, output_columns, slow_time, shutdown):
    def scrape_settings(connection, rows):
        result = []
        with connection.cursor(row_factory=dict_row) as cursor:
            tns = time.time_ns() / 1000
            cursor.execute("SHOW ALL;")
            for record in cursor:
                setting_name = record["name"]
                if setting_name in rows:
                    setting_type = rows[setting_name]
                    setting_str = record["setting"]
                    result.append((setting_name, _parse_field(setting_type, setting_str)))

        connection.commit()
        result.append(("time", tns))
        result.sort(key=lambda t: t[0])
        return result

    def scrape_table(connection, query):
        # Open a cursor to perform database operations
        tuples = []
        columns = []
        with connection.cursor() as cursor:
            # Query the database and obtain data as Python objects.
            cursor.execute(query, prepare=False)
            binary = []
            for i, column in enumerate(cursor.description):
                if column.type_code == 17:
                    binary.append(i)
                columns.append(column.name)

            for record in cursor:
                rec = list(record)
                for binary_col in binary:
                    rec[binary_col] = binascii.hexlify(record[binary_col])
                tuples.append(rec)

        connection.commit()
        return columns, tuples

    setproctitle.setproctitle("Userspace Collector Process")
    with psycopg.connect("host=localhost port=5432 dbname=benchbase user=wz2", autocommit=True) as connection:
        # Poll on the Collector's output buffer until Collector is shut down.
        while not shutdown.is_set():
            try:
                knob_values = scrape_settings(connection, KNOBS)
                knob_columns = [k[0] for k in knob_values]
                knob_values = [k[1] for k in knob_values]
                output_columns["pg_settings"] = knob_columns
                output_rows["pg_settings"].append(knob_values)

                for target, query in PG_COLLECTOR_TARGETS.items():
                    columns, tuples = scrape_table(connection, query)
                    output_columns[target] = columns
                    output_rows[target].extend(tuples)

                time.sleep(slow_time)
            except KeyboardInterrupt:
                logger.info("Userspace Collector caught KeyboardInterrupt.")
            except Exception as e:  # pylint: disable=broad-except
                # TODO(Matt): If postgres shuts down the connection closes and we get an exception for that.
                logger.warning("Userspace Collector caught %s.", e)

    logger.info("Userspace Collector shut down.")


def main(benchmark, outdir, collector_interval):
    keep_running = True

    # Augment with pgstattuple_approx data.
    tables = BENCHDB_TO_TABLES[benchmark]
    for tbl in tables:
        PG_COLLECTOR_TARGETS[tbl] = f"SELECT EXTRACT(epoch from NOW())*1000000 as time, * FROM pgstattuple_approx('{tbl}');"

    with mp.Manager() as manager:
        # Create coordination data structures for Collectors and Processors
        collector_flags = manager.dict()
        collector_processes = {}

        shutdown = manager.Event()

        pg_scrape_columns = manager.dict()
        pg_scrape_tuples = manager.dict()
        pg_scrape_tuples["pg_settings"] = manager.list()
        for target, _ in PG_COLLECTOR_TARGETS.items():
            pg_scrape_tuples[target] = manager.list()

        pg_collector_process = mp.Process(
            target=pg_collector,
            args=(
                pg_scrape_tuples,
                pg_scrape_columns,
                collector_interval,
                shutdown,
            ),
        )
        pg_collector_process.start()

        while keep_running:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                keep_running = False
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Collector caught %s.", e)

        print("Collector shutting down.")

        # Shut down the Collectors so that
        # no more data is generated for the Processors.
        shutdown.set()

        pg_collector_process.join()

        PG_COLLECTOR_TARGETS["pg_settings"] = None
        for target in PG_COLLECTOR_TARGETS.keys():
            file_path = f"{outdir}/{target}.csv"
            write_header = not Path(file_path).exists()
            with open(file_path, "a", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(pg_scrape_columns[target])
                writer.writerows(pg_scrape_tuples[target])
        print("Collector wrote out pg collector data.")

        # We're done.
        sys.exit()


class CollectorCLI(cli.Application):
    benchmark = cli.SwitchAttr(
        "--benchmark",
        str,
        mandatory=True,
        help="Benchmark that is being executed.",
    )

    collector_interval = cli.SwitchAttr(
        "--collector_interval",
        int,
        mandatory=False,
        default=60,
        help="TIme between pg collector invocations.",
    )

    outdir = cli.SwitchAttr(
        "--outdir",
        str,
        mandatory=False,
        default=".",
        help="Training data output directory",
    )

    def main(self):
        main(self.benchmark, self.outdir, self.collector_interval)


if __name__ == "__main__":
    Collector.run()
