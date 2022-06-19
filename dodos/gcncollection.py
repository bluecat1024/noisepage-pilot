import os

import doit
from plumbum import cmd

def task_enable_pg_autoexplain():
    """
    Set up the pg_autoexplain to be json format, printed to the log.
    """
    sql_list1 = [
        # Note that this will overwrite any existing settings of shared_preload_libraries.
        "ALTER SYSTEM SET shared_preload_libraries='auto_explain'",
    ]

    sql_list2 = [
        "ALTER SYSTEM SET auto_explain.log_min_duration=0",
        "ALTER SYSTEM SET auto_explain.log_analyze='on'",
        "ALTER SYSTEM SET auto_explain.log_format='json'",
        "ALTER SYSTEM SET auto_explain.sample_rate=1",
        "ALTER SYSTEM SET log_statement='none'",
    ]

    def enable_pg_autoexplain(dbname, dbuser, dbpass):
        for sql in sql_list1:
            os.system(f'PGPASSWORD={dbpass} psql -h localhost --dbname={dbname} --username={dbuser} --command="{sql}"')
        cmd.sudo["systemctl"]["restart", "postgresql"].run_fg()
        for sql in sql_list2:
            os.system(f'PGPASSWORD={dbpass} psql -h localhost --dbname={dbname} --username={dbuser} --command="{sql}"')


    return {
        "actions": [
            enable_pg_autoexplain,
            lambda: cmd.sudo["systemctl"]["restart", "postgresql"].run_fg(),
            "until pg_isready ; do sleep 1 ; done",
        ],
        "uptodate": [False],
        "params": [
            {
                "name": "dbname",
                "long": "dbname",
                "help": "Postgres DB name.",
                "default": "benchbase_gcn",
            },
            {
                "name": "dbuser",
                "long": "dbuser",
                "help": "Superuser name.",
                "default": "su_gcn",
            },
            {
                "name": "dbpass",
                "long": "dbpass",
                "help": "Superuser pass.",
                "default": "pass_gcn",
            },
        ],
    }

def task_disable_pg_autoexplain():
    sql_list = [
        "ALTER SYSTEM SET auto_explain.sample_rate=0",
    ]

    def disable_pg_autoexplain(dbname, dbuser, dbpass):
        for sql in sql_list:
            os.system(f'PGPASSWORD={dbpass} psql -h localhost --dbname={dbname} --username={dbuser} --command="{sql}"')

    return {
        "actions": [
            disable_pg_autoexplain,
            lambda: cmd.sudo["systemctl"]["restart", "postgresql"].run_fg(),
            "until pg_isready ; do sleep 1 ; done",
        ],
        "uptodate": [False],
        "params": [
            {
                "name": "dbname",
                "long": "dbname",
                "help": "Postgres DB name.",
                "default": "benchbase_gcn",
            },
            {
                "name": "dbuser",
                "long": "dbuser",
                "help": "Superuser name.",
                "default": "su_gcn",
            },
            {
                "name": "dbpass",
                "long": "dbpass",
                "help": "Superuser pass.",
                "default": "pass_gcn",
            },
        ],
    }