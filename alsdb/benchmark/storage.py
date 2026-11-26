"""Storage conversion utilities for benchmark data."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _process_peak_rss_mb(pid: int) -> float | None:
    """Return the process' peak resident memory in MiB when available.

    Linux exposes the high-water mark as ``VmHWM``. On macOS, ``ps`` provides
    a sampled resident-set size; the caller polls this function while PDAL runs.
    """
    status_file = Path(f"/proc/{pid}/status")
    if status_file.exists():
        try:
            for line in status_file.read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
        except (OSError, ValueError, IndexError):
            return None
        return None

    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        return float(value) / 1024.0 if value else None
    except (OSError, ValueError):
        return None


class LAZToCOPCConverter:
    """Convert LAZ tiles to COPC, skipping completed outputs."""

    def __init__(self, input_dir: str | Path, output_dir: str | Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input directory does not exist: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "conversion_log.csv"

    def _log(self, laz_file: Path, status: str, elapsed: float, peak_mb: float | None) -> None:
        """Append one conversion record to the persistent CSV log."""
        write_header = not self.log_file.exists() or self.log_file.stat().st_size == 0
        with self.log_file.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if write_header:
                writer.writerow(
                    ["timestamp", "input_file", "output_file", "status", "seconds", "peak_rss_mb"]
                )
            writer.writerow(
                [
                    datetime.now().astimezone().isoformat(timespec="seconds"),
                    str(laz_file),
                    str(self.output_dir / f"{laz_file.stem}.copc.laz"),
                    status,
                    f"{elapsed:.3f}",
                    "" if peak_mb is None else f"{peak_mb:.1f}",
                ]
            )

    def convert_file(self, laz_file: Path) -> str:
        """Convert one LAZ file; return ``converted``, ``skipped`` or ``failed``."""
        copc_file = self.output_dir / f"{laz_file.stem}.copc.laz"
        if copc_file.is_file():
            print(f"Skipping {laz_file.name}: output already exists")
            self._log(laz_file, "skipped", 0.0, None)
            return "skipped"

        # Keep .copc.laz as the final suffix so PDAL recognises the output format.
        temp_file = self.output_dir / f".{laz_file.stem}.{os.getpid()}.{time.time_ns()}.tmp.copc.laz"
        print(f"Converting {laz_file.name}")
        started = time.perf_counter()
        process: subprocess.Popen | None = None
        peak_mb: float | None = None
        status = "failed"

        try:
            process = subprocess.Popen(
                ["pdal", "translate", str(laz_file), str(temp_file)]
            )
            while process.poll() is None:
                measured = _process_peak_rss_mb(process.pid)
                if measured is not None:
                    peak_mb = measured if peak_mb is None else max(peak_mb, measured)
                time.sleep(0.1)

            return_code = process.wait()
            elapsed = time.perf_counter() - started
            if return_code != 0:
                raise RuntimeError(f"PDAL exited with status {return_code}")
            if not temp_file.is_file() or temp_file.stat().st_size == 0:
                raise RuntimeError("PDAL did not produce a non-empty output file")

            # Atomic on the same filesystem: the final name appears only when complete.
            os.replace(temp_file, copc_file)
            status = "converted"
            print(
                f"  done in {elapsed:.1f} s; peak RSS "
                f"{peak_mb:.1f} MiB" if peak_mb is not None
                else f"  done in {elapsed:.1f} s; peak RSS unavailable"
            )
        except KeyboardInterrupt:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        except (OSError, RuntimeError) as exc:
            elapsed = time.perf_counter() - started
            print(f"  ERROR: {exc}", file=sys.stderr)
        finally:
            if temp_file.exists():
                temp_file.unlink()

        self._log(laz_file, status, elapsed, peak_mb)
        return status

    def convert(self) -> int:
        """Convert all top-level LAZ files and return a process exit code."""
        laz_files = sorted(self.input_dir.glob("*.laz"))
        print(f"Found {len(laz_files)} LAZ files in {self.input_dir}")
        if not laz_files:
            print("No .laz files found", file=sys.stderr)
            return 1

        started = time.perf_counter()
        counts = {"converted": 0, "skipped": 0, "failed": 0}
        try:
            for index, laz_file in enumerate(laz_files, 1):
                print(f"[{index}/{len(laz_files)}] ", end="", flush=True)
                counts[self.convert_file(laz_file)] += 1
        except KeyboardInterrupt:
            print("\nConversion interrupted; incomplete temporary output was removed.")
            return 130

        elapsed = time.perf_counter() - started
        print("\nConversion finished")
        print(f"Converted: {counts['converted']}")
        print(f"Skipped:   {counts['skipped']}")
        print(f"Failed:    {counts['failed']}")
        print(f"Total time: {elapsed / 60:.2f} minutes")
        print(f"Log: {self.log_file}")
        return 1 if counts["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    """Run storage-related benchmark commands."""
    parser = argparse.ArgumentParser(prog="python -m alsdb.benchmark storage")
    subparsers = parser.add_subparsers(dest="storage_command", required=True)
    convert_parser = subparsers.add_parser(
        "convert", help="convert each LAZ tile to a COPC file"
    )
    convert_parser.add_argument("laz_dir", type=Path, help="folder containing LAZ tiles")
    convert_parser.add_argument("copc_dir", type=Path, help="folder for COPC outputs")
    args = parser.parse_args(argv)

    if args.storage_command == "convert":
        return LAZToCOPCConverter(args.laz_dir, args.copc_dir).convert()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
