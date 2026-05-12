# SysForge Workflow Project Report

## 1. Overview

SysForge is a workflow-based GPU measurement and performance-analysis tool. It is designed to evaluate CUDA-capable systems by collecting hardware characteristics, running generated CUDA probes, profiling workloads with Nsight Compute, validating the collected results, and exporting a structured machine-readable report.

The profiling workflow is built to support tasks where hardware information cannot always be obtained from a single static device query. Instead, SysForge actively measures properties such as memory latency, cache behavior, shared-memory performance, global-memory bandwidth, boost clock, and selected profiler metrics. It combines LLM-driven code generation with compiler execution, runtime testing, profiler integration, and validation logic.

The primary entrypoint is `python -m sysforge.main profiling`, which writes its final output to `output.json`, making the system suitable for automated evaluation pipelines, benchmarking environments, and reproducible GPU-analysis workflows.

---

## 2. Primary Features

### 2.1 Autonomous GPU Probe Generation

One of the main features of SysForge is its ability to automatically generate CUDA microbenchmarks for requested hardware targets. For a supported measurement target, the profiling workflow creates CUDA source code that is intended to measure the requested property directly on the available GPU.

Examples of supported probe targets include:

- L1 latency
- L2 latency
- DRAM latency
- L2 cache capacity
- Shared-memory peak bandwidth
- Global-memory peak bandwidth
- Actual boost clock
- Bank-conflict penalty
- Maximum shared memory per block
- Visible SM count

This allows the workflow to go beyond simple device-property lookup and instead perform active measurement.

---

### 2.2 Target-Aware Measurement Strategy

SysForge uses a catalog of known hardware targets. Each catalog entry defines the expected unit, measurement strategy, plausible value range, and measurement description.

This feature helps the workflow select a more appropriate method for each target. For example:

- DRAM latency is measured through a large working-set pointer chase.
- L2 cache capacity is inferred through a latency-vs-working-set-size sweep.
- SM count is inferred by counting distinct `%smid` values.
- Shared-memory bandwidth is measured using block-local shared-memory operations.
- Actual boost clock is measured from device clock cycles and elapsed time.

By encoding target-specific expectations, the system reduces the chance of using a generic or incorrect benchmark for specialized GPU measurements.

---

### 2.3 LLM-Guided Self-Repair

The profiling workflow includes a self-repair loop for generated CUDA programs. If a generated probe fails to compile, crashes at runtime, produces no extractable result, or returns an implausible value, SysForge can request a corrected version from the LLM.

The repair process handles several failure categories:

- Compilation errors
- Runtime failures
- Missing or malformed output
- Implausible measurement values
- Target-specific sanity-check failures

This makes the system more robust than a one-shot code-generation workflow. The workflow can iteratively improve the measurement program based on actual compiler and runtime feedback.

---

### 2.4 CUDA Compilation and Execution Automation

SysForge automatically writes generated CUDA source files into the workspace, compiles them with `nvcc`, and executes the resulting binaries. It manages build paths, source paths, binary paths, execution timeouts, and error capture.

This automation allows each measurement to proceed through a complete local execution pipeline:

1. Generate CUDA code.
2. Save source file.
3. Compile with `nvcc`.
4. Run the binary.
5. Capture stdout and stderr.
6. Extract the result.
7. Validate the result.
8. Store the final measurement.

The user does not need to manually compile or run each benchmark.

---

### 2.5 Nsight Compute Metric Support

SysForge also supports Nsight Compute metric targets. Metric names that follow the Nsight Compute-style pattern are routed to the profiling path instead of the CUDA probe-generation path.

For these targets, the profiling workflow generates a reference GEMM workload, compiles it, runs it, profiles it with `ncu`, and parses the CSV output. This enables direct collection of metrics such as SM throughput, memory throughput, device attributes, and other Nsight Compute counters.

This feature is important because some requested values are not best measured with custom CUDA code. They are more accurately obtained through the profiler.

---

### 2.6 Per-Metric Aggregation

When Nsight Compute returns multiple rows for a metric, SysForge aggregates the values into a stable result. Numeric metric samples are collected, and the median value is used as the representative result.

For each metric, the output may include:

- Final value
- Unit
- Number of samples
- Minimum sample value
- Maximum sample value
- Raw sample values
- Error information if the metric is missing

This makes profiler-based measurements easier to consume programmatically.

---

### 2.7 Bottleneck Analysis and Recommendations

For profiler-based runs, SysForge can summarize performance bottlenecks and provide optimization recommendations. The workflow uses collected Nsight Compute metrics to produce a higher-level analysis that may include:

- Bottleneck classification
- Supporting evidence
- Optimization suggestions
- Human-readable summary

This feature turns raw profiler counters into a more useful performance-analysis report.

---

## 3. Validation Features

### 3.1 Plausibility Checking

SysForge validates probe results against expected ranges defined in the target catalog. If a value is outside the plausible range, the workflow does not immediately accept it. Instead, it can request a corrected probe or return a low-confidence best-effort value only after retry options are exhausted.

This prevents obviously incorrect results from being silently reported as successful measurements.

---

### 3.2 Target-Specific Sanity Checks

The project includes specialized sanity rules for common GPU-benchmarking mistakes. These checks improve the reliability of generated benchmarks by detecting cases where the code appears to run but is likely measuring the wrong thing.

Examples of issues the workflow can detect include:

- DRAM latency probes that accidentally measure cache hits.
- Pointer-chase benchmarks with insufficient working-set size.
- Cache-capacity sweeps that do not show a valid latency cliff.
- Shared-memory bandwidth probes that actually measure global memory.
- Bank-conflict tests that fail to create real bank conflicts.
- SM-count probes that report block count instead of distinct SM IDs.
- Clock measurements that exceed realistic boost-clock hints.
- Bandwidth tests with too few or too many iterations per thread.

These checks are a major feature of the project because they address the practical difficulty of generating correct GPU microbenchmarks.

---

### 3.3 Stability Through Re-Runs

When a probe produces an accepted value, SysForge can rerun the accepted binary to collect additional samples. The final result is selected from accepted samples using the median.

This improves stability and reduces dependence on a single execution, which is especially useful for GPU measurements that may vary due to scheduling, clock behavior, thermal state, or system load.

---

### 3.4 Confidence Reporting

Each hardware result includes a confidence value. Successful probe results receive confidence based on the extraction result and corroborating samples. Nsight Compute metric results receive high confidence when the metric is successfully captured and low confidence when missing.

This gives downstream consumers a way to distinguish strong measurements from partial or best-effort outputs.

---

## 4. Reporting Features

### 4.1 Structured JSON Output

SysForge writes results to `output.json`. The output is designed for automated scoring, inspection, and downstream processing.

The output includes:

- Requested input targets
- Routed probe targets
- Routed profiler metrics
- Hardware measurement results
- Nsight Compute analysis results
- Environment hints
- Execution trace
- Errors
- Start and finish timestamps

This structure makes the report both machine-readable and audit-friendly.

---

### 4.2 Unified Hardware Result Format

Probe-based results and Nsight Compute-based results are promoted into the same top-level `hardware` dictionary. Each target is keyed by name and includes fields such as value, unit, confidence, source, and error status.

This unified format simplifies consumption by evaluators or external tools because all requested target results can be read from a single location.

---

### 4.3 Execution Traceability

SysForge records detailed trace information for probe attempts. The trace can include:

- Probe version
- Generation or repair phase
- Source path
- Compilation status
- Compiler stderr
- Runtime status
- Runtime stdout and stderr tails
- Extracted values
- Plausibility status
- Rejection reason
- Agent rationale

This trace is useful for debugging, auditing, and understanding why a particular value was accepted or rejected.

---

### 4.4 Partial Output Flushing

After each completed probe target, SysForge writes partial results to the output file. This improves resilience during long-running measurements. If the process stops before all targets finish, completed results may still be available.

---

## 5. Workflow Entry Points

SysForge now exposes explicit workflow subcommands instead of a single implicit orchestration path:

- `python -m sysforge.main profiling`
- `python -m sysforge.main optimize-lora`

Today, `profiling` is the implemented production workflow, while `optimize-lora` is a registered scaffold for future work.

