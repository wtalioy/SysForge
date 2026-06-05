# SysForge Workflow Report

## 1. Summary

SysForge is a workflow-driven GPU systems tool. It runs local measurement, profiling, and optimization loops, validates the result, and writes structured artifacts for later inspection.

The implemented workflows are:

- hardware profiling
- inference-runtime optimization
- LoRA extension optimization

The important design point is that SysForge does not treat an LLM response or generated artifact as the final answer. Each workflow closes the loop with local execution, validation, measurement, and structured reporting.

## 2. Workflow Map

| Workflow | Main Goal | Primary Artifact | Core Evidence |
| --- | --- | --- | --- |
| Hardware profiling | Measure GPU hardware behavior and profiler metrics | structured hardware report | probe outputs, Nsight Compute metrics, plausibility checks |
| Runtime optimization | Improve an LLM inference runtime implementation | promoted runtime artifact | reference-model correctness, robust throughput benchmarks, promotion trace |
| LoRA optimization | Optimize a LoRA-style CUDA extension | promoted extension artifact | correctness checks, benchmark tiers, final confirmation |

## 3. Profiling

The profiling workflow answers GPU hardware and performance questions through active measurement.

It supports generated CUDA probes for targets such as:

- L1, L2, and DRAM latency
- L2 capacity
- global and shared-memory bandwidth
- boost clock
- bank-conflict penalty
- shared memory per block
- visible SM count

It also routes Nsight Compute metric requests into a profiler-backed analysis path. For those requests, SysForge builds a reference workload, runs Nsight Compute, parses metric rows, aggregates samples, and produces a higher-level bottleneck summary when possible.

The distinctive value of this workflow is target-aware measurement. It does not use one generic benchmark for every question; it selects probe logic, expected units, and plausibility checks based on the requested target.

## 4. Optimize-Runtime

The runtime-optimization workflow improves an LLM inference runtime while keeping the implementation space bounded and measurable.

### 4.1 Candidate Generation

Candidates are rendered from a fixed runtime template. The LLM does not write arbitrary runtime code. Instead, it selects schema-validated strategy choices from supported template knobs, including:

- prefill batching policy
- KV-cache allocation and growth policy
- attention policy
- cache layout policy
- normalization policy

The workflow starts from initial baseline strategies, writes a bootstrap runtime artifact, evaluates all candidates, then asks the LLM for additional high-quality catalogue strategies using measured evidence.

### 4.2 Mandatory Correctness

Every candidate must pass one comprehensive correctness check before it can be considered. The check compares runtime logits against a reference model across scenarios that include:

- prefill and decode
- mixed request lengths
- request removal
- replacement prefill
- reordered request IDs
- cache reuse

There is no separate basic/stress toggle. The broad correctness check is the only correctness gate.

### 4.3 Mandatory Benchmarking

Every candidate must also pass robust benchmarking. The benchmark suite covers:

- prefill
- decode
- mixed traffic
- varied prefill lengths
- long decode
- request churn

Benchmark runs are repeated, warmup runs are discarded, and summaries use stable aggregate values such as medians and spread percentages.

### 4.4 Promotion

Promotion requires both correctness and benchmark evidence. Candidate ranking prioritizes:

- mixed throughput
- churn throughput
- long-decode decode throughput
- decode throughput
- varied-prefill throughput
- lower peak memory

Promotion guards reject small noisy gains when robust-case regressions are present. After promotion, the workflow reruns correctness and benchmarking on the promoted runtime artifact. If either final check fails, the workflow fails instead of silently keeping the artifact.

## 5. Optimize-LoRA

The LoRA-optimization workflow optimizes a LoRA-style CUDA extension through candidate-family search.

### 5.1 Candidate Families

The LLM proposes parameterized candidate families rather than isolated kernels. A family can provide a template plus parameters, or a curated set of concrete variants. SysForge expands those variants locally, builds them, benchmarks them, and records outcomes.

This makes the LLM useful for search-space design while leaving the winner decision to local evidence.

### 5.2 Staged Evaluation

The workflow uses staged benchmarking:

- early screening filters weak variants quickly
- fuller tiers evaluate shortlisted candidates
- final confirmation reruns the strongest finalists

This controls search cost while still grounding promotion in stronger measurements.

### 5.3 Control Logic

The controller tracks the incumbent, challenger quality, close frontiers, stalled rounds, family revision, regeneration, and fallback recovery. It can keep the incumbent when a challenger regresses, defer early stop when results are close, and revise future families using measured feedback.

Promising candidates can also be profiled with Nsight Compute so the workflow can attach bottleneck hints to candidate records.

## 6. Reliability Guarantees

SysForge uses different validation logic per workflow, but the contract is consistent: outputs should be backed by local evidence.

- Profiling checks measured values against target-specific plausibility rules.
- Runtime optimization requires reference-model correctness and robust benchmark evidence before promotion.
- LoRA optimization requires correctness and benchmark confirmation before accepting an optimized artifact.

Repeated measurements reduce sensitivity to GPU noise. Structured errors, logs, traces, and candidate records make failed or partial runs inspectable.
