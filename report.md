# SysForge Workflow Project Report

## 1. Overview

SysForge is a workflow-driven GPU systems tool for measurement, profiling, and code optimization on CUDA-capable machines. Instead of exposing a single monolithic pipeline, SysForge provides multiple workflow entry points that share the same runtime model: generate or transform workload code, execute it locally, validate the result, and emit a structured `output.json` artifact.

Today the project includes two implemented workflows:

- `python -m sysforge.main profiling`
- `python -m sysforge.main optimize-lora`

The `profiling` workflow focuses on answering hardware and performance questions through active measurement and profiler-driven analysis. The `optimize-lora` workflow focuses on improving a LoRA-style CUDA extension through LLM-authored candidate families, local benchmarking, promotion logic, and final confirmation.

The key idea across both workflows is the same: SysForge uses agentic control to close the loop between code generation, compilation, execution, validation, and structured reporting.

---

## 2. Overall System Features

### 2.1 Workflow-Based Architecture

SysForge is organized around explicit workflow subcommands rather than a single hard-coded path. Each workflow is registered independently, uses a shared runtime context, and returns a typed result object that is serialized to `output.json`.

This design has a few practical benefits:

- different tasks can use different control logic without forking the whole system
- shared infrastructure can be reused across workflows
- new workflows can be added without rewriting the CLI contract
- outputs remain machine-readable even when workflow internals differ

---

### 2.2 Agentic Closed-Loop Execution

Both workflows follow a closed-loop execution model instead of a one-shot prompt model. SysForge does not stop at generating code or proposing an idea. It compiles artifacts, runs them, checks results, records failures, and uses the observed outcome to decide what to do next.

Across the project, this loop may include:

- code generation or code transformation
- local compilation with `nvcc` or extension build tooling
- runtime execution and benchmarking
- profiler collection through Nsight Compute
- result validation and plausibility checks
- retry, repair, revision, or promotion decisions

This is one of the main strengths of the project: the LLM is part of a measured system, not the sole source of truth.

---

### 2.3 Local Build and Execution Automation

SysForge automates the full local execution path for generated or optimized CUDA artifacts. It manages source files, build directories, binaries or Python extensions, subprocess execution, timeouts, and error capture.

That automation is important in both workflows:

- `profiling` uses it to compile and run generated CUDA probes or profiler workloads
- `optimize-lora` uses it to compile candidate kernel variants and benchmark them on a fixed harness

The result is a reproducible local pipeline that can be used in benchmarking and evaluation environments without manual intervention at each step.

---

### 2.4 Structured, Audit-Friendly Results

Every workflow writes a structured `output.json` file. The exact schema differs by workflow, but the shared design goal is the same: make the output useful for both automated scoring and human inspection.

Depending on the workflow, the output may include:

- workflow identity and status
- start and finish timestamps
- environment hints
- measured results or benchmark summaries
- candidate artifacts
- validation outcomes
- trace or controller history
- errors and notes

This makes SysForge suitable not only for running experiments, but also for reviewing how a result was produced.

---

## 3. Profiling Workflow Features

### 3.1 Autonomous GPU Probe Generation

The `profiling` workflow can generate CUDA microbenchmarks for requested hardware targets rather than relying only on static device queries.

Examples of supported probe targets include:

- L1 latency
- L2 latency
- DRAM latency
- L2 cache capacity
- shared-memory peak bandwidth
- global-memory peak bandwidth
- actual boost clock
- bank-conflict penalty
- maximum shared memory per block
- visible SM count

This allows SysForge to actively measure hardware behavior that is often unavailable or unreliable through simple metadata lookup.

---

### 3.2 Target-Aware Measurement Strategy

SysForge uses a target catalog that defines expected units, measurement strategies, plausible value ranges, and target descriptions. The profiling workflow uses this catalog to select more appropriate measurement logic for each requested target.

For example:

- DRAM latency uses a large working-set pointer chase
- L2 cache capacity is inferred from a latency sweep
- SM count is inferred from distinct `%smid` values
- boost clock is estimated from device cycles and elapsed time

Encoding target-specific expectations reduces the chance of measuring the wrong phenomenon with a generic benchmark.

---

### 3.3 Nsight Compute Metric Collection

The profiling workflow also supports Nsight Compute metrics. Metric names that match the profiler-style format are routed into an analysis path rather than probe synthesis.

For these targets, SysForge:

1. builds a reference workload
2. runs Nsight Compute
3. parses the resulting metric data
4. aggregates repeated samples into stable outputs

This makes the profiling workflow useful for both generated probes and profiler-derived counters.

---

### 3.4 Bottleneck Analysis

When profiler metrics are available, SysForge can summarize likely bottlenecks and provide optimization-oriented observations. Rather than exposing only raw counters, it can produce a higher-level analysis with evidence and recommendations.

This makes the profiling workflow useful as a performance-analysis tool, not just a metric collection script.

---

## 4. Optimize-LoRA Workflow Features

### 4.1 Baseline Bootstrap and Search Initialization

The `optimize-lora` workflow starts from a baseline LoRA-style CUDA extension and treats that implementation as the verified incumbent. SysForge writes the baseline artifact, validates it on benchmark shapes, and then uses it as the reference point for search.

This gives the workflow a stable starting artifact and ensures that every optimized candidate is compared against a concrete local baseline.

---

### 4.2 LLM-Authored Candidate Families

Instead of asking the model for a single kernel implementation, SysForge asks for parameterized candidate families. A family can define a source template plus a small parameter space, or an explicit curated set of concrete variants.

This is an important project feature because it combines:

- LLM-guided search-space design
- local enumeration of variants
- measured comparison instead of purely textual ranking

The workflow can also revise a family in later rounds using round feedback, recent history, and incumbent context.

---

### 4.3 Multi-Stage Benchmarking and Promotion

Candidate search uses staged evaluation rather than a single benchmark pass. SysForge screens candidates on lighter benchmark tiers, promotes strong variants to fuller evaluation, and finally confirms the best candidates with a more stable final pass.

At a high level, the workflow uses:

- early screening to filter weak candidates quickly
- full evaluation for shortlisted candidates
- final confirmation for the strongest finalists

This staged structure helps control search cost while still making the winner decision based on stronger evidence.

---

### 4.4 Search Control and Recovery

The optimize-lora workflow includes explicit control logic for promotion, close-frontier handling, stalled rounds, family revision, regeneration, and fallback recovery.

Examples of behavior the controller supports include:

- keeping the incumbent when a challenger regresses
- deferring early stop when the frontier is too close to call
- revising a family after round feedback
- regenerating a fresh family when revision gets boxed into duplicate history
- falling back to local search patterns if LLM-generated continuation fails

This makes the workflow more robust than a naive “generate N kernels and pick the fastest” script.

---

### 4.5 Optional Profile-Guided Candidate Inspection

The optimize-lora workflow can optionally profile promising candidates and summarize their bottlenecks. This reuses SysForge’s profiler integration in service of kernel search rather than standalone hardware reporting.

That is a good example of the project’s shared architecture: profiler support is not confined to one workflow, even though the workflows use it for different goals.

---

## 5. Validation and Reliability Features

### 5.1 Plausibility and Correctness Checks

SysForge does not treat a completed run as automatically correct. Each workflow includes validation logic appropriate to its task.

- `profiling` validates measured values against plausible ranges and target-specific sanity rules
- `optimize-lora` validates candidate correctness and compares benchmark behavior against the incumbent and reference

This protects the system from accepting outputs that compiled successfully but do not mean what they appear to mean.

---

### 5.2 Self-Repair and Retry Paths

A central feature of SysForge is that failures feed back into the control loop. Depending on the workflow, the system can recover from:

- compile failures
- runtime failures
- malformed or missing output
- implausible measurements
- duplicate or low-value candidate revisions

In the profiling workflow, this appears as LLM-guided probe repair. In the optimize-lora workflow, it appears as family revision, regeneration, reranking, reruns, and fallback logic.

---

### 5.3 Stability Through Re-Runs

SysForge uses repeated measurements to reduce sensitivity to noise. The exact mechanism varies by workflow, but the general pattern is consistent: rerun important evaluations, aggregate samples, and prefer stable summary statistics such as medians or geometric means.

This matters because GPU measurements can vary due to scheduling, clock behavior, thermal conditions, or transient system load.

---

## 6. Reporting Features

### 6.1 Machine-Readable JSON Output

Each workflow emits an `output.json` file tailored to its task.

For `profiling`, the output emphasizes:

- requested targets
- routed probe and metric targets
- hardware results
- analysis summaries
- per-target trace information

For `optimize-lora`, the output emphasizes:

- candidate records
- benchmark tier summaries
- current best artifact
- controller trace
- winner confirmation state

Despite those differences, both workflows preserve the same reporting principles: explicit status, structured fields, and enough traceability to reconstruct what happened.

---

### 6.2 Traceability and Debuggability

SysForge records detailed intermediate state so that results can be audited after the fact.

Examples include:

- probe attempt traces in `profiling`
- controller and promotion traces in `optimize-lora`
- compile and runtime errors
- benchmark summaries
- accepted or rejected outcomes

This is valuable both for debugging the system itself and for understanding why a final result should be trusted.

---

### 6.3 Partial Progress Preservation

SysForge writes intermediate progress so that long-running workflows are more resilient to interruption.

This is especially useful when:

- the profiling workflow is measuring many targets
- the optimize-lora workflow is compiling and benchmarking many candidates

Even when a full run does not complete perfectly, partial progress may still be available for inspection.

---

## 7. Workflow Entry Points

SysForge currently exposes these workflow entry points:

- `python -m sysforge.main profiling`
- `python -m sysforge.main optimize-lora`

Together, these workflows show that SysForge is not just a profiler wrapper and not just a kernel search script. It is a general workflow platform for GPU-facing agentic tasks, with shared execution infrastructure and task-specific control logic layered on top.
