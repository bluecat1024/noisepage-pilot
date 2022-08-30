# NoisePage Pilot

This repository contains the pilot components for the [NoisePage DBMS](https://noise.page/).

## Quickstart

1. Init all submodules `git submodule update --init --recursive`.
2. Install necessary packages.
    - `cd behavior/modeling/featurewiz && pip3 install --upgrade -r requirements.txt`
    - `pip3 install --upgrade -r requirements.txt`
3. List all the tasks.
    - `doit list`
4. Select and run a doit task from the task list, e.g. `doit action_recommendation`.  Task dependencies are executed automatically.

## Background

- Self-Driving DBMS = Workload Forecasting + Behavior Modeling + Action Planning.
    - Workload Forecasting: `forecast` folder.
    - Modeling: WIP.
    - Action Planning: `action` folder.
    - See [^electricsheep], [^15799] for more details.

## References

[^electricsheep]: Make Your Database System Dream of Electric Sheep: Towards Self-Driving Operation.

    ```
    @article{pavlo21,
    author = {Pavlo, Andrew and Butrovich, Matthew and Ma, Lin and Lim, Wan Shen and Menon, Prashanth and Van Aken, Dana and Zhang, William},
    title = {Make Your Database System Dream of Electric Sheep: Towards Self-Driving Operation},
    journal = {Proc. {VLDB} Endow.},
    volume = {14},
    number = {12},
    pages = {3211--3221},
    year = {2021},
    url = {https://db.cs.cmu.edu/papers/2021/p3211-pavlo.pdf},
    }
    ```

[^15799]: https://15799.courses.cs.cmu.edu/spring2022/
