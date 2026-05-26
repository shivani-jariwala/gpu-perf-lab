"""Command-line entrypoint: the `gpu-perf` command.

Phase 0 ships the plumbing commands only:

    gpu-perf --version
    gpu-perf info                 # resolved config + the experiment matrix that WILL run

Later phases add:
    gpu-perf launch ...           # Phase 1: Slurm (or torchrun fallback) launcher
    gpu-perf train ...            # Phase 2: instrumented DDP training run
    gpu-perf nccl ...             # Phase 3: NCCL microbenchmarks
    gpu-perf report ...           # Phase 4: telemetry correlation + comparison report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from . import __version__
from .config import Config, load_config
from .models import Backend, NcclOp, ScalingMode
from .launch.detect import probe_backend
from .launch.launcher import DEFAULT_TARGET, PROBE_TARGET, build_plan, submit
from .nccl.parse import busbw_factor
from .nccl.run import build_command, find_binary, parse_file, run_nccl
from .report import charts as charts_mod
from .report.render import (analyze, build_report_dict, load_records,
                            render_console, render_markdown)


def _load_or_die(config_path: str | None) -> Config:
    try:
        return load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)


def _find_experiment(cfg: Config, label: str):
    for rc in cfg.experiments():
        if rc.label == label:
            return rc
    labels = ", ".join(rc.label for rc in cfg.experiments())
    click.echo(f"error: no experiment labelled '{label}'. Available: {labels}", err=True)
    sys.exit(1)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="gpu-perf")
def main() -> None:
    """Distributed GPU Performance Lab (single-node, up to 2 GPUs)."""


@main.command()
@click.option("--config", "config_path", default=None, help="Path to experiments.yaml")
def info(config_path: str | None) -> None:
    """Show the resolved configuration and the experiment matrix that WILL run."""
    cfg = _load_or_die(config_path)

    click.echo(f"gpu-perf {__version__}")
    click.echo(f"config source : {cfg.source_path}")
    click.echo("")
    click.echo("Fixed across all runs:")
    click.echo(f"  model      : {cfg.model}")
    click.echo(f"  dataset    : {cfg.dataset}")
    click.echo(f"  precision  : {cfg.precision}")
    click.echo(f"  steps      : {cfg.steps} (+{cfg.warmup_steps} warmup)")
    click.echo("")

    hw = cfg.hardware
    click.echo("Assumed hardware (SIMULATED sizing — confirm & re-baseline):")
    click.echo(f"  gpu_model    : {cfg.gpu_model}")
    click.echo(f"  gpu_count    : {hw.get('gpu_count', '?')}")
    click.echo(f"  vram_gb      : {hw.get('vram_gb', '?')}")
    click.echo(f"  interconnect : {hw.get('interconnect', '?')}")
    click.echo("")

    st = probe_backend(prefer_slurm=True)
    parts = f" partitions=[{st.slurm_partitions}]" if st.slurm_partitions else ""
    click.echo(f"Launcher backend : {st.chosen.value}  ({st.reason()})"
               f"{parts}")
    click.echo(f"  torchrun on PATH: {st.torchrun_available}")
    click.echo("")

    tel = cfg.telemetry
    click.echo(f"Telemetry source : {tel.get('source', '(unset)')} "
               f"(interval {tel.get('sample_interval_ms', '?')} ms)")
    ncfg = cfg.nccl
    click.echo(f"NCCL ops         : {', '.join(ncfg.get('ops', [])) or '(none)'}  "
               f"[busbw floor {ncfg.get('min_busbw_gbps', '?')} GB/s — provisional]")
    click.echo("")

    click.echo("Experiment matrix (model + dataset fixed; num_gpus + batch vary):")
    click.echo(f"  {'label':<16} {'mode':<7} {'gpus':>4} {'global_bs':>10} {'per_gpu_bs':>11}")
    click.echo(f"  {'-'*16} {'-'*7} {'-'*4} {'-'*10} {'-'*11}")
    try:
        for rc in cfg.experiments():
            click.echo(f"  {rc.label:<16} {rc.scaling_mode.value:<7} {rc.num_gpus:>4} "
                       f"{rc.global_batch_size:>10} {rc.per_gpu_batch_size:>11}")
    except ValueError as exc:
        click.echo(f"  error building matrix: {exc}", err=True)
        sys.exit(1)

    modes = {rc.scaling_mode for rc in cfg.experiments()}
    have = ", ".join(sorted(m.value for m in modes if m is not ScalingMode.SINGLE))
    click.echo("")
    click.echo(f"Scaling comparisons available: {have or '(none)'} "
               f"(2 GPUs => illustrative comparison, not a large sweep)")


@main.command()
@click.argument("label")
@click.option("--config", "config_path", default=None, help="Path to experiments.yaml")
@click.option("--prefer-slurm/--no-prefer-slurm", default=True,
              help="Use Slurm if available (default) or force torchrun.")
@click.option("--target", default=None,
              help=f"Per-process entrypoint module (default: {DEFAULT_TARGET}).")
@click.option("--probe", "use_probe", is_flag=True, default=False,
              help="Launch the built-in distributed smoke target instead of the trainer.")
@click.option("--run", "do_run", is_flag=True, default=False,
              help="Actually submit/execute (default: dry-run, just print the plan).")
def launch(config_path, label, prefer_slurm, target, use_probe, do_run) -> None:
    """Build the launch command for an experiment (Slurm, else torchrun).

    Defaults to DRY-RUN: prints the chosen backend, why, and the exact command
    (or rendered sbatch script). Pass --run to actually submit/execute.

        gpu-perf launch strong-2gpu                 # dry-run, show the plan
        gpu-perf launch strong-2gpu --probe --run   # real smoke test on the GPU box
    """
    cfg = _load_or_die(config_path)
    rc = _find_experiment(cfg, label)
    entry = PROBE_TARGET if use_probe else (target or DEFAULT_TARGET)

    # The trainer resolves its own config from experiments.yaml via --label
    # (single source of truth); the probe takes no args.
    if use_probe:
        target_args: list[str] = []
    else:
        target_args = ["--label", label]
        if config_path:
            target_args += ["--config", config_path]

    plan = build_plan(rc, target=entry, target_args=target_args, prefer_slurm=prefer_slurm)

    click.echo(f"Experiment : {rc.label}  "
               f"(mode={rc.scaling_mode.value} gpus={rc.num_gpus} "
               f"global_bs={rc.global_batch_size} per_gpu_bs={rc.per_gpu_batch_size})")
    click.echo(f"Backend    : {plan.backend.value}  ({plan.reason})")
    click.echo(f"Entrypoint : {entry}")
    for n in plan.notes:
        click.echo(f"  note: {n}")

    if plan.backend is Backend.SLURM:
        click.echo("\nRendered sbatch script:")
        click.echo("-" * 60)
        click.echo(plan.sbatch_text.rstrip())
        click.echo("-" * 60)
        click.echo(f"Submit with: sbatch <written to evidence/logs/ on --run>")
    else:
        click.echo(f"\nCommand: {plan.describe()}")

    if not do_run:
        click.echo("\n[dry-run] nothing executed. Re-run with --run on the GPU box.")
        return

    click.echo("\n[run] executing ...")
    code = submit(plan)
    if plan.sbatch_path:
        click.echo(f"sbatch script written to {plan.sbatch_path}")
    click.echo(f"exit code: {code}")
    sys.exit(code)


def _print_nccl_table(parse) -> None:
    click.echo(f"  {'size(B)':>12} {'time(us)':>10} {'algbw(GB/s)':>12} {'busbw(GB/s)':>12} {'#wrong':>7}")
    click.echo(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*7}")
    for r in parse.rows:
        click.echo(f"  {r.size_bytes:>12} {r.time_us:>10.2f} {r.algbw_gbps:>12.3f} "
                   f"{r.busbw_gbps:>12.3f} {r.wrong_count:>7}")
    f = busbw_factor(parse.op, parse.num_gpus) if parse.num_gpus else 0.0
    click.echo(f"\n  op={parse.op.value} gpus={parse.num_gpus} busbw_factor={f:.3f} "
               f"(busbw = algbw x factor)")
    click.echo(f"  peak busbw = {parse.peak_busbw_gbps:.2f} GB/s   "
               f"avg busbw = {parse.avg_busbw_gbps}   total #wrong = {parse.total_wrong}")


@main.command()
@click.option("--op", type=click.Choice(["all_reduce", "all_gather", "broadcast"]),
              default="all_reduce", help="NCCL collective to benchmark.")
@click.option("--gpus", "num_gpus", type=int, default=2, help="Number of GPUs (-g).")
@click.option("--config", "config_path", default=None, help="Path to experiments.yaml")
@click.option("--tests-dir", default=None, help="Path to a built nccl-tests dir.")
@click.option("--parse", "parse_path", default=None,
              help="Parse an existing nccl-tests log instead of running (works on any machine).")
@click.option("--run", "do_run", is_flag=True, default=False,
              help="Actually execute the binary (lab box only). Default: dry-run.")
@click.option("--out", default=None, help="Where to write the RunRecord JSON.")
def nccl(op, num_gpus, config_path, tests_dir, parse_path, do_run, out) -> None:
    """NCCL microbenchmark: run (or parse) all_reduce_perf / all_gather_perf.

        gpu-perf nccl --op all_reduce --gpus 2            # dry-run: show the command
        gpu-perf nccl --op all_reduce --gpus 2 --run      # execute on the lab box
        gpu-perf nccl --parse evidence/samples/nccl-...log --op all_reduce --gpus 2
    """
    op_enum = NcclOp(op)

    # --- parse an existing log (no GPU needed) -------------------------------
    if parse_path:
        record, parse = parse_file(parse_path, op=op_enum, num_gpus=num_gpus)
        click.echo(f"Parsed {parse_path}  {'[SIMULATED]' if record.simulated else ''}")
        _print_nccl_table(parse)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
            click.echo(f"\nRunRecord written to {out}")
        return

    # --- pull sweep params from config if present ----------------------------
    min_b, max_b, step = "8", "512M", 2
    if config_path or Path("config/experiments.yaml").exists():
        try:
            ncfg = _load_or_die(config_path).nccl
            min_b = str(ncfg.get("min_bytes", min_b))
            max_b = str(ncfg.get("max_bytes", max_b))
            step = int(ncfg.get("step_factor", step))
        except SystemExit:
            pass

    cmd = build_command(op_enum, num_gpus=num_gpus, min_bytes=min_b,
                        max_bytes=max_b, step_factor=step)
    binary = find_binary(op_enum, tests_dir)
    click.echo(f"Collective : {op}   gpus={num_gpus}   sweep {min_b}..{max_b} x{step}")
    click.echo(f"Binary     : {binary or '(not found — build nccl-tests / set $NCCL_TESTS_DIR)'}")
    click.echo(f"Command    : {' '.join(cmd)}")

    if not do_run:
        click.echo("\n[dry-run] nothing executed. Re-run with --run on the lab box "
                   "(or use --parse on a captured log).")
        return

    if not binary:
        click.echo("error: binary not found; cannot --run here.", err=True)
        sys.exit(1)

    click.echo("\n[run] executing nccl-tests ...")
    record, raw = run_nccl(op_enum, num_gpus=num_gpus, min_bytes=min_b,
                           max_bytes=max_b, step_factor=step, tests_dir=tests_dir)
    log_path = Path(f"evidence/logs/nccl-{op}-{num_gpus}gpu.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(raw, encoding="utf-8")
    out_path = Path(out or f"evidence/logs/nccl-{op}-{num_gpus}gpu.json")
    out_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    from .nccl.parse import parse_nccl_output
    _print_nccl_table(parse_nccl_output(raw, op=op_enum, num_gpus=num_gpus))
    click.echo(f"\nraw log -> {log_path}\nRunRecord -> {out_path}")


@main.command()
@click.option("--inputs", "-i", multiple=True,
              help="RunRecord JSON files or globs (repeatable). "
                   "Default: evidence/logs/*.json + evidence/samples/*.json")
@click.option("--config", "config_path", default=None, help="Path to experiments.yaml (for thresholds)")
@click.option("--out-dir", default="evidence/reports", help="Where to write report.json / report.md")
@click.option("--charts/--no-charts", default=True, help="Also render PNG charts (needs [viz]).")
@click.option("--chart-dir", default="evidence/charts", help="Where to write chart PNGs.")
def report(inputs, config_path, out_dir, charts, chart_dir) -> None:
    """Correlate timings with telemetry, classify bottlenecks, compare configs.

        gpu-perf report                         # default inputs -> console + files
        gpu-perf report -i "evidence/samples/*.json"
    """
    input_globs = list(inputs) or ["evidence/logs/*.json", "evidence/samples/*.json"]
    records = load_records(input_globs)
    if not records:
        click.echo(f"error: no RunRecord JSONs found in: {', '.join(input_globs)}", err=True)
        sys.exit(1)

    # Optional threshold overrides from config.
    thresholds = None
    try:
        thresholds = _load_or_die(config_path).thresholds if (
            config_path or Path("config/experiments.yaml").exists()) else None
    except SystemExit:
        thresholds = None

    training, nccl, comparisons = analyze(records, thresholds)
    click.echo(render_console(training, nccl, comparisons))

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(build_report_dict(training, nccl, comparisons), indent=2), encoding="utf-8")
    (out / "report.md").write_text(
        render_markdown(training, nccl, comparisons), encoding="utf-8")
    click.echo(f"Wrote {out/'report.json'} and {out/'report.md'}")

    if charts:
        if not charts_mod.charts_available():
            click.echo("charts: matplotlib not installed — skipping "
                       "(install with pip install -e \".[viz]\").")
        else:
            written = charts_mod.generate_charts(training, nccl, comparisons, chart_dir)
            click.echo(f"charts: wrote {len(written)} PNG(s) to {chart_dir}")


if __name__ == "__main__":
    main()
