# Behavior Modeling

This document details the core components of behavior modeling and how to use them.

## Postgres Plan Nodes

The following is a list of Postgres query plan nodes.

- Agg
- Append
- CteScan
- CustomScan
- ForeignScan
- FunctionScan
- Gather
- GatherMerge
- Group
- HashJoinImpl
- IncrementalSort
- IndexOnlyScan
- IndexScan
- Limit
- LockRows
- Material
- MergeAppend
- MergeJoin
- ModifyTable
- NamedTuplestoreScan
- NestLoop
- ProjectSet
- RecursiveUnion
- Result
- SampleScan
- SeqScan
- SetOp
- Sort
- SubPlan
- SubqueryScan
- TableFuncScan
- TidScan
- Unique
- ValuesScan
- WindowAgg
- WorkTableScan

## BenchBase Benchmark Databases

The following BenchBase benchmarks have been tested to work with behavior modeling.

- AuctionMark
- Epinions
- SEATS
- SIBench
- SmallBank
- TATP
- TPC-C
- Twitter
- Voter
- Wikipedia
- YCSB

Caveats:

- TPC-H support is blocked on the [native loader](https://github.com/cmu-db/benchbase/pull/99) being merged.
- Epinions is missing results for the Materialize OU in the plan generated for `GetReviewsByUser`.
    - `SELECT * FROM review r, useracct u WHERE u.u_id = r.u_id AND r.u_id=$1 ORDER BY rating LIMIT 10`

## Resource Consumption Metrics

We currently only gather elapsed_us from the QSS extension.

## Operating Unit (OU) Model Variants

- Tree-based
    - dt
    - rf - good performance
    - gbm - good performance
- Multi-layer perceptron
    - mlp
- Generalized linear models
    - lr
    - huber
    - mt_lasso
    - lasso
    - mt_elastic
    - elastic

## References

See [^mb2] for more details.

[^mb2]: MB2: Decomposed Behavior Modeling for Self-Driving Database Management Systems

    ```
    @article{ma21,
    author = {Ma, Lin and Zhang, William and Jiao, Jie and Wang, Wuwen and Butrovich, Matthew and Lim, Wan Shen and Menon, Prashanth and Pavlo, Andrew},
    title = {MB2: Decomposed Behavior Modeling for Self-Driving Database Management Systems},
    journal = {SIGMOD},
    year = {2021},
    url = {https://www.cs.cmu.edu/~malin199/publications/2021.mb2.sigmod.pdf},
    }
    ```
