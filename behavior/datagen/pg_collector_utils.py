import re
from enum import Enum, auto, unique
from distutils import util


# Type of the pg_setting variable.
@unique
class SettingType(Enum):
    BOOLEAN = auto()
    INTEGER = auto()
    BYTES = auto()
    INTEGER_TIME = auto()
    FLOAT_TIME = auto()
    FLOAT = auto()


# Convert a string time unit to milliseconds.
def _time_unit_to_ms(str):
    if str == "d":
        return 1000 * 60 * 60 * 24
    elif str == "h":
        return 1000 * 60 * 60
    elif str == "min":
        return 1000 * 60
    elif str == "s":
        return 1000
    elif str == "ms":
        return 1
    elif str == "us":
        return 1.0 / 1000
    else:
        return None


# Parse a pg_setting field value.
def _parse_field(type, value):
    if type == SettingType.BOOLEAN:
        return util.strtobool(value)
    elif type == SettingType.INTEGER:
        return int(value)
    elif type == SettingType.BYTES:
        if value in ["-1", "0"]:
            # Hardcoded default/disabled values for this field.
            return int(value)
        bytes_regex = re.compile(r"(\d+)\s*([kmgtp]?b)", re.IGNORECASE)
        order = ("b", "kb", "mb", "gb", "tb", "pb")
        field_bytes = None
        for number, unit in bytes_regex.findall(value):
            field_bytes = int(number) * (1024 ** order.index(unit.lower()))
        assert field_bytes is not None, f"Failed to parse bytes from value string {value}"
        return field_bytes
    elif type == SettingType.INTEGER_TIME:
        if value == "-1":
            # Hardcoded default/disabled values for this field.
            return int(value)
        bytes_regex = re.compile(r"(\d+)\s*((?:d|h|min|s|ms|us)?)", re.IGNORECASE)
        field_ms = None
        for number, unit in bytes_regex.findall(value):
            field_ms = int(number) * _time_unit_to_ms(unit)
        assert field_ms is not None, f"Failed to parse time from value string {value}"
        return field_ms
    elif type == SettingType.FLOAT_TIME:
        if value == "0":
            # Hardcoded default/disabled values for this field.
            return int(value)
        bytes_regex = re.compile(r"(\d+(?:\.\d+)?)\s*((?:d|h|min|s|ms|us)?)", re.IGNORECASE)
        field_ms = None
        for number, unit in bytes_regex.findall(value):
            field_ms = float(number) * _time_unit_to_ms(unit)
        assert field_ms is not None, f"Failed to parse time from value string {value}"
        return field_ms
    elif type == SettingType.FLOAT:
        return float(value)
    else:
        return None

KNOBS = {
    # https://www.postgresql.org/docs/current/runtime-config-autovacuum.html
    "autovacuum": SettingType.BOOLEAN,
    "autovacuum_max_workers": SettingType.INTEGER,
    "autovacuum_naptime": SettingType.INTEGER_TIME,
    "autovacuum_vacuum_threshold": SettingType.INTEGER,
    "autovacuum_vacuum_insert_threshold": SettingType.INTEGER,
    "autovacuum_analyze_threshold": SettingType.INTEGER,
    "autovacuum_vacuum_scale_factor": SettingType.FLOAT,
    "autovacuum_vacuum_insert_scale_factor": SettingType.FLOAT,
    "autovacuum_analyze_scale_factor": SettingType.FLOAT,
    "autovacuum_freeze_max_age": SettingType.INTEGER,
    "autovacuum_multixact_freeze_max_age": SettingType.INTEGER,
    "autovacuum_vacuum_cost_delay": SettingType.FLOAT_TIME,
    "autovacuum_vacuum_cost_limit": SettingType.INTEGER,
    # https://www.postgresql.org/docs/12/runtime-config-resource.html
    "maintenance_work_mem": SettingType.BYTES,
    "autovacuum_work_mem": SettingType.BYTES,
    "vacuum_cost_delay": SettingType.FLOAT_TIME,
    "vacuum_cost_page_hit": SettingType.INTEGER,
    "vacuum_cost_page_miss": SettingType.INTEGER,
    "vacuum_cost_page_dirty": SettingType.INTEGER,
    "vacuum_cost_limit": SettingType.INTEGER,
    "effective_io_concurrency": SettingType.INTEGER,
    "maintenance_io_concurrency": SettingType.INTEGER,
    "max_worker_processes": SettingType.INTEGER,
    "max_parallel_workers_per_gather": SettingType.INTEGER,
    "max_parallel_maintenance_workers": SettingType.INTEGER,
    "max_parallel_workers": SettingType.INTEGER,

    "jit": SettingType.BOOLEAN,
    "hash_mem_multiplier": SettingType.FLOAT,
    "effective_cache_size": SettingType.BYTES,
    "shared_buffers": SettingType.BYTES,
}
