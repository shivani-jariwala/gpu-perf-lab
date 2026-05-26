"""Phase 3 — NCCL microbenchmarks (nccl-tests).

Wraps and parses the standard nccl-tests binaries (`all_reduce_perf`,
`all_gather_perf`) into ``NcclResult`` rows: per-size collective latency,
algorithm bandwidth (algbw), and bus bandwidth (busbw). busbw is the
hardware-comparable number — see ``parse.busbw_factor`` for why.
"""
