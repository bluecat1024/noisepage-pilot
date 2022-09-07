#!/usr/bin/env bash

export DB_USER="su_gcn"
export DB_PASS="pass_gcn"
export DB_NAME="benchbase_gcn"
export PG_PATH="artifacts/noisepage"


# Setup the database using the global constants.
_setup_database() {
  # Drop the project database if it exists.
  PGPASSWORD=${DB_PASS} dropdb --host=localhost --username=${DB_USER} --if-exists ${DB_NAME}
  # Create the project database.
  PGPASSWORD=${DB_PASS} createdb --host=localhost --username=${DB_USER} ${DB_NAME}
}

_setup_benchmark() {
  benchmark="${1}"
  scalefactor="${2}"

  echo "Loading: ${benchmark} SF= ${scalefactor}"

  # Modify the BenchBase benchmark configuration.
  mkdir -p artifacts/project/
  cp ./config/behavior/benchbase/${benchmark}_config.xml ./artifacts/project/${benchmark}_config.xml
  xmlstarlet edit --inplace --update '/parameters/url' --value "jdbc:postgresql://localhost:5432/${DB_NAME}?preferQueryMode=simple" ./artifacts/project/${benchmark}_config.xml
  xmlstarlet edit --inplace --update '/parameters/username' --value "${DB_USER}" ./artifacts/project/${benchmark}_config.xml
  xmlstarlet edit --inplace --update '/parameters/password' --value "${DB_PASS}" ./artifacts/project/${benchmark}_config.xml
  xmlstarlet edit --inplace --update '/parameters/scalefactor' --value "${scalefactor}" ./artifacts/project/${benchmark}_config.xml
  xmlstarlet edit --inplace --update '/parameters/works/work/time' --value "30" ./artifacts/project/${benchmark}_config.xml
  xmlstarlet edit --inplace --update '/parameters/works/work/rate' --value "unlimited" ./artifacts/project/${benchmark}_config.xml

  # Load the benchmark into the project database.
  doit benchbase_run --benchmark="${benchmark}" --config="./artifacts/project/${benchmark}_config.xml" --args="--create=true --load=true"
}

_change_exec_conf() {
    benchmark="${1}"
    nterminal="${2}"
    xmlstarlet edit --inplace --update '/parameters/terminals' --value ${nterminal} ./artifacts/project/${benchmark}_config.xml
}

_dump_database() {
  dump_path="${1}"

  # Dump the project database into directory format.
  rm -rf "./${dump_path}"
  PGPASSWORD=$DB_PASS "${PG_PATH}/pg_dump" --host=localhost --username=$DB_USER --format=directory --file=./${dump_path} $DB_NAME

  echo "Dumped database to: ${dump_path}"
}

_restore_database() {
  dump_path="${1}"

  # Restore the project database from directory format.
  PGPASSWORD=${DB_PASS} "${PG_PATH}/pg_restore" --host=localhost --username=$DB_USER --clean --if-exists --dbname=${DB_NAME} ./${dump_path}

  echo "Restored database from: ${dump_path}"
}

_clear_log_folder() {
  sudo bash -c "rm -rf ${PG_PATH}/pgdata/log/*"
  echo "Cleared all query logs."
}

_copy_logs() {
  save_path="${1}"

  # TODO(WAN): Is there a way to ensure all flushed?
  sleep 10
  sudo bash -c "cat ${PG_PATH}/pgdata/log/*.csv > ${save_path}"
  echo "Copied all query logs to: ${save_path}"
}

kill_descendant_processes() {
  local pid="$1"
  local and_self="${2:-false}"
  if children="$(pgrep -P "$pid")"; then
    for child in $children; do
      kill_descendant_processes "$child" true
    done
  fi
  if [[ "$and_self" == true ]]; then
    sudo kill -9 "$pid"
  fi
}

exit_cleanly() {
  kill_descendant_processes $$
}

benchmark="${1}"
scalefactor="${2}"
nterminal="${3}"
createnewdb="${4}"

trap exit_cleanly SIGINT
trap exit_cleanly SIGTERM

set -e
# xmlstarlet to edit BenchBase XML configurations.
sudo apt-get -qq install xmlstarlet

benchmark_dump_folder="./artifacts/project/dumps"
# Create the folder for all the benchmark dumps.
mkdir -p "./${benchmark_dump_folder}"

doit noisepage_init --dbname="benchbase_gcn"

benchmark_dump_path="./${benchmark_dump_folder}/${benchmark}_primary"
if [ "${createnewdb}" == "1" ]; then
    _setup_database 
    _setup_benchmark "${benchmark}" "${scalefactor}"

    _dump_database "${benchmark_dump_path}"
else
    _restore_database "${benchmark_dump_path}"
fi

_change_exec_conf "${benchmark}" "${nterminal}"

doit noisepage_enable_logging
#doit enable_pg_autoexplain
doit noisepage_qss_install
doit benchbase_run --benchmark="${benchmark}" --config="./artifacts/project/${benchmark}_config.xml" --args="--execute=true"
doit noisepage_disable_logging

workload_csv_folder="./artifacts/project/${benchmark}_${scalefactor}_${nterminal}"
mkdir -p ${workload_csv_folder}
workload_csv="${workload_csv_folder}/workload.csv"
_copy_logs "${workload_csv}"
# _clear_log_folder


