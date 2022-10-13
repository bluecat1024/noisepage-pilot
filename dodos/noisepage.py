import time
import os
from pathlib import Path

import doit
from doit.action import CmdAction
from plumbum import local

from dodos import VERBOSITY_DEFAULT, default_artifacts_path, default_build_path

ARTIFACTS_PATH = default_artifacts_path()
BUILD_PATH = default_build_path()
DEFAULT_POSTGRESQL_CONF_PATH = Path("config/postgres/default_postgresql.conf").absolute()

DEFAULT_DB = "benchbase"
DEFAULT_USER = "admin"
DEFAULT_PASS = "pass_gcn"
DEFAULT_PGDATA = "pgdata"

# Output: various useful binaries and folders.
ARTIFACT_createdb = ARTIFACTS_PATH / "createdb"
ARTIFACT_createuser = ARTIFACTS_PATH / "createuser"
ARTIFACT_dropdb = ARTIFACTS_PATH / "dropdb"
ARTIFACT_dropuser = ARTIFACTS_PATH / "dropuser"
ARTIFACT_pg_config = ARTIFACTS_PATH / "pg_config"
ARTIFACT_pg_ctl = ARTIFACTS_PATH / "pg_ctl"
ARTIFACT_postgres = ARTIFACTS_PATH / "postgres"
ARTIFACT_psql = ARTIFACTS_PATH / "psql"
ARTIFACT_pgdata = ARTIFACTS_PATH / DEFAULT_PGDATA
ARTIFACT_pgdata_log = ARTIFACTS_PATH / DEFAULT_PGDATA / "log"


def task_noisepage_clone():
    """
    NoisePage: clone.
    """

    def repo_clone(repo_url):
        cmd = f"git clone {repo_url} --branch new_pg14_gcn --single-branch --depth 1 {BUILD_PATH}"
        return cmd

    return {
        "actions": [
            # Set up directories.
            f"mkdir -p {BUILD_PATH}",
            # Clone NoisePage.
            CmdAction(repo_clone),
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "targets": [BUILD_PATH, BUILD_PATH / "Makefile"],
        "uptodate": [True],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "repo_url",
                "long": "repo_url",
                "help": "The repository to clone from.",
                "default": "https://github.com/bluecat1024/postgres.git",
            },
        ],
    }


def task_noisepage_build():
    """
    NoisePage: build.
    """
    return {
        "actions": [
            lambda: os.chdir(BUILD_PATH),
            # Configure NoisePage.
            "doit np_config --build_type=release",
            # Compile NoisePage.
            "doit np_build",
            # Move artifacts out.
            lambda: os.chdir(doit.get_initial_workdir()),
            f"mkdir -p {ARTIFACTS_PATH}",
            f"cp {BUILD_PATH / 'build/bin/*'} {ARTIFACTS_PATH}",
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [BUILD_PATH / "Makefile"],
        "targets": [
            ARTIFACTS_PATH,
            ARTIFACT_createdb,
            ARTIFACT_createuser,
            ARTIFACT_dropdb,
            ARTIFACT_dropuser,
            ARTIFACT_pg_config,
            ARTIFACT_pg_ctl,
            ARTIFACT_postgres,
            ARTIFACT_psql,
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_noisepage_init():
    """
    NoisePage: start a clean NoisePage instance in detached mode.
    """

    def run_noisepage_detached(config, dbname):
        assert Path(config).exists(), f"{config} does not exist."
        local["cp"][f"{config}", f"{DEFAULT_PGDATA}/postgresql.conf"].run_nohup()
        ret = local["./pg_ctl"]["start", "-D", DEFAULT_PGDATA].run_nohup(stdout="noisepage.out")
        print(f"NoisePage PID: {ret.pid}")

    sql_list = [
        f"CREATE ROLE {DEFAULT_USER} WITH LOGIN SUPERUSER ENCRYPTED PASSWORD '{DEFAULT_PASS}'",
    ]

    return {
        "actions": [
            "pkill postgres || true",
            lambda: os.chdir(ARTIFACTS_PATH),
            f"rm -rf {DEFAULT_PGDATA}",
            f"./initdb {DEFAULT_PGDATA}",
            run_noisepage_detached,
            "until ./pg_isready ; do sleep 1 ; done",
            f"./createdb {DEFAULT_DB}",
            *[f'./psql --dbname={DEFAULT_DB} --command="{sql}"' for sql in sql_list],
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [ARTIFACT_pg_ctl],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "config",
                "long": "config",
                "help": "Path to the postgresql.conf configuration file.",
                "default": DEFAULT_POSTGRESQL_CONF_PATH,
            },
            {
                "name": "dbname",
                "long": "dbname",
                "help": "Database name",
                "default": DEFAULT_DB,
            },
        ],
    }


def task_noisepage_stop():
    """
    NoisePage: stop NoisePage instance.
    """

    def stop(data):
        while len(list(local.pgrep("postgres"))) != 0:
            local["./pg_ctl"]["stop", "-D", data].run_fg(retcode=None)
            time.sleep(5)

    return {
        "actions": [
            lambda: os.chdir(ARTIFACTS_PATH),
            stop,
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [ARTIFACT_pg_ctl],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "data",
                "long": "data",
                "help": "Postgres Data Directory location",
                "default": DEFAULT_PGDATA,
            },
        ],
    }

def task_noisepage_swap_config():
    """
    NoisePage: swaps the postgresql.conf for the current running instance.
    """

    def swap_config(config):
        local["./pg_ctl"]["stop", "-D", DEFAULT_PGDATA, "-m", "smart"].run(retcode=None)
        local["cp"][f"{config}", f"{DEFAULT_PGDATA}/postgresql.conf"].run_nohup()
        ret = local["./pg_ctl"]["start", "-D", DEFAULT_PGDATA].run_nohup(stdout="noisepage.out")
        print(f"NoisePage PID: {ret.pid}")

    return {
        "actions": [
            lambda: os.chdir(ARTIFACTS_PATH),
            swap_config,
            "until ./pg_isready ; do sleep 1 ; done",
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "file_dep": [ARTIFACT_pg_ctl],
        "uptodate": [False],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "config",
                "long": "config",
                "help": "Path to the postgresql.conf configuration file.",
                "default": DEFAULT_POSTGRESQL_CONF_PATH,
            },
        ],
    }


def task_noisepage_qss_install():
    """
    NoisePage: install qss extension to enable query state collection.
    """
    sql_list1 = [
        "ALTER SYSTEM SET allow_system_table_mods = ON",
    ]

    sql_list2 = [
        "DROP TABLE IF EXISTS pg_catalog.pg_qss_plans",
        "DROP TABLE IF EXISTS pg_catalog.pg_qss_stats",
        "DROP TABLE IF EXISTS pg_catalog.pg_qss_ddl",
        """CREATE UNLOGGED TABLE pg_catalog.pg_qss_plans(
            query_id bigint,
            generation integer,
            db_id integer,
            pid integer,
            statement_timestamp bigint,
            features text,
            primary key(query_id, generation, db_id, pid)
            )
            WITH (autovacuum_enabled = OFF)""",
        """CREATE UNLOGGED TABLE pg_catalog.pg_qss_stats(
            query_id bigint,
	    db_id integer,
	    pid integer,
	    statement_timestamp bigint,
	    plan_node_id int,
            elapsed_us float8,
	    counter0 float8,
	    counter1 float8,
	    counter2 float8,
	    counter3 float8,
	    counter4 float8,
	    counter5 float8,
	    counter6 float8,
	    counter7 float8,
	    counter8 float8,
	    counter9 float8,
            payload bigint,
            txn bigint,
            comment text
            )
            WITH (autovacuum_enabled = OFF)""",
        """CREATE UNLOGGED TABLE pg_catalog.pg_qss_ddl(
	    db_id integer,
	    statement_timestamp bigint,
            query text,
            command text
            )
            WITH (autovacuum_enabled = OFF)""",
        "ALTER SYSTEM SET shared_preload_libraries='qss','pgstattuple'",
        "ALTER SYSTEM SET qss_capture_enabled = ON",
        "ALTER SYSTEM SET qss_capture_nested = OFF",
        "ALTER SYSTEM SET qss_capture_exec_stats = ON",
        "ALTER SYSTEM SET qss_output_noisepage = OFF",
        "ALTER SYSTEM SET log_statement='none'",
        "DROP EXTENSION IF EXISTS qss",
        "DROP EXTENSION IF EXISTS pgstattuple",
        "CREATE EXTENSION qss",
        "CREATE EXTENSION pgstattuple",
    ]

    def run_query_in_db(dbname):
        for sql in sql_list1:
            os.system(f'PGPASSWORD={DEFAULT_PASS} ./psql --dbname={dbname} --username={DEFAULT_USER} --command="{sql}"')
        local["./pg_ctl"]["restart", "-D", DEFAULT_PGDATA, "-m", "smart"].run_fg(retcode=None)
        for sql in sql_list2:
            os.system(f'PGPASSWORD={DEFAULT_PASS} ./psql --dbname={dbname} --username={DEFAULT_USER} --command="{sql}"')
        local["./pg_ctl"]["restart", "-D", DEFAULT_PGDATA, "-m", "smart"].run_fg(retcode=None)


    return {
        "actions": [
            lambda: os.chdir(BUILD_PATH),
            # Compile and install qss.
            "doit qss_install",
            # Alter system table to allow modification.
            lambda: os.chdir(ARTIFACTS_PATH),
            run_query_in_db,
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "uptodate": [False],
        "file_dep": [ARTIFACT_postgres],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "dbname",
                "long": "dbname",
                "help": "The database name where to install qss.",
                "default": DEFAULT_DB,
            },
        ],
    }

def task_noisepage_tscout_decouple():
    """
    NoisePage: install hutch extension to support EXPLAIN (format tscout).
    """
    sql_list = [
        # Note that this will overwrite any existing settings of shared_preload_libraries.
        "ALTER SYSTEM SET tscout_feature_decoupled = on",
    ]

    return {
        "actions": [
            lambda: os.chdir(ARTIFACTS_PATH),
            *[
                f'PGPASSWORD={DEFAULT_PASS} ./psql --dbname={DEFAULT_DB} --username={DEFAULT_USER} --command="{sql}"'
                for sql in sql_list
            ],
            lambda: local["./pg_ctl"]["restart", "-D", DEFAULT_PGDATA].run_fg(),
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "uptodate": [False],
        "file_dep": [ARTIFACT_postgres],
        "verbosity": VERBOSITY_DEFAULT,
    }

def task_noisepage_qcache_install():
    """
    NoisePage: install qcache extension to enable serialized
    query store for a certain database.
    """

    sql_list1 = [
        "ALTER SYSTEM SET allow_system_table_mods = ON",
    ]

    sql_list2 = [
        "DROP TABLE IF EXISTS pg_catalog.pg_qcache",
        """CREATE TABLE pg_catalog.pg_qcache(
            queryid bigint,
            dbid integer,
            procid integer,
            timestamp bigint,
            features text,
            primary key(queryid, dbid, procid)
            )""",
        "ALTER SYSTEM SET shared_preload_libraries='qcache'",
    ]

    def run_query_in_db(dbname):
        for sql in sql_list1:
            os.system(f'PGPASSWORD={DEFAULT_PASS} ./psql --dbname={dbname} --username={DEFAULT_USER} --command="{sql}"')
        local["./pg_ctl"]["restart", "-D", DEFAULT_PGDATA].run_fg()
        for sql in sql_list2:
            os.system(f'PGPASSWORD={DEFAULT_PASS} ./psql --dbname={dbname} --username={DEFAULT_USER} --command="{sql}"')
        local["./pg_ctl"]["restart", "-D", DEFAULT_PGDATA].run_fg()


    return {
        "actions": [
            lambda: os.chdir(BUILD_PATH),
            # Compile and install qcache.
            "doit qcache_install",
            # Alter system table to allow modification.
            lambda: os.chdir(ARTIFACTS_PATH),
            run_query_in_db,
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "uptodate": [False],
        "file_dep": [ARTIFACT_postgres],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "dbname",
                "long": "dbname",
                "help": "The database name where to install qcache.",
                "default": DEFAULT_DB,
            },
        ],
    }


def task_noisepage_enable_logging():
    """
    NoisePage: enable logging. (will cause a restart)
    """
    sql_list = [
        "ALTER SYSTEM SET log_destination='csvlog'",
        "ALTER SYSTEM SET log_statement='all'",
        "ALTER SYSTEM SET logging_collector=on",
    ]

    return {
        "actions": [
            lambda: os.chdir(ARTIFACTS_PATH),
            *[
                f'PGPASSWORD={DEFAULT_PASS} ./psql --dbname={DEFAULT_DB} --username={DEFAULT_USER} --command="{sql}"'
                for sql in sql_list
            ],
            lambda: local["./pg_ctl"]["restart", "-D", DEFAULT_PGDATA, "-m", "smart"].run_fg(),
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }


def task_noisepage_disable_logging():
    """
    NoisePage: disable logging. (will cause a restart)
    """
    sql_list = [
        "ALTER SYSTEM SET log_destination='stderr'",
        "ALTER SYSTEM SET log_statement='none'",
        "ALTER SYSTEM SET logging_collector=off",
    ]

    return {
        "actions": [
            lambda: os.chdir(ARTIFACTS_PATH),
            *[
                f'PGPASSWORD={DEFAULT_PASS} ./psql --dbname={DEFAULT_DB} --username={DEFAULT_USER} --command="{sql}"'
                for sql in sql_list
            ],
            lambda: local["./pg_ctl"]["restart", "-D", DEFAULT_PGDATA, "-m", "smart"].run_fg(),
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "verbosity": VERBOSITY_DEFAULT,
    }

def task_noisepage_set_fillfactor():
    """
    NoisePage: Set fill factor of all tables and VACUUM FULL.
    Not supporting other workloads than TPC-C.
    """
    table_list = ['warehouse', 'district', 'item', 'customer', 'stock',\
        'oorder', 'new_order', 'history', 'order_line',]

    def run_query_in_db(fillfactor):
        for tname in table_list:
            sql = f"ALTER TABLE {tname} SET ( fillfactor = {fillfactor})"
            os.system(f'PGPASSWORD={DEFAULT_PASS} ./psql --dbname={DEFAULT_DB} --username={DEFAULT_USER} --command="{sql}"')
            sql = f"VACUUM FULL {tname}"
            os.system(f'PGPASSWORD={DEFAULT_PASS} ./psql --dbname={DEFAULT_DB} --username={DEFAULT_USER} --command="{sql}"')

    return {
        "actions": [
            lambda: os.chdir(ARTIFACTS_PATH),
            run_query_in_db,
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "verbosity": VERBOSITY_DEFAULT,
        "params": [
            {
                "name": "fillfactor",
                "long": "fillfactor",
                "help": "Specify the fillfactor.",
                "default": DEFAULT_DB,
            },
        ],
    }


def task_noisepage_truncate_log():
    """
    NoisePage: truncate the most recent query log files.
    """

    return {
        "actions": [
            lambda: os.chdir(ARTIFACT_pgdata_log),
            'RECENT=$(ls --sort=time --reverse *.csv | head --lines=%(num_files)s) ; echo "Deleting:\n$RECENT" ; rm $RECENT ; echo "Deleted."',
            # Reset working directory.
            lambda: os.chdir(doit.get_initial_workdir()),
        ],
        "verbosity": VERBOSITY_DEFAULT,
        "uptodate": [False],
        "params": [
            {
                "name": "num_files",
                "long": "num_files",
                "help": "The number of query log files to remove.",
                "default": 5,
            },
        ],
    }
