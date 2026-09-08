"""Benchmark suites for Evolutionary Algorithms."""
from .base import BenchmarkFunction
from .cec_suite import get_benchmark_suite, CECBenchmark

__all__ = ["BenchmarkFunction", "get_benchmark_suite", "CECBenchmark"]
