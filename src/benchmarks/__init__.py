"""Benchmark suites for Evolutionary Algorithms."""
from benchmarks.base import BenchmarkFunction
from benchmarks.cec_suite import get_benchmark_suite, CECBenchmark

__all__ = ["BenchmarkFunction", "get_benchmark_suite", "CECBenchmark"]
