from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import strftime

from ..core.runtime import RuntimeContext
from .base import Workflow, WorkflowResult


BASELINE_SOURCE = """#include <torch/extension.h>

namespace {

void check_inputs(
    const torch::Tensor& W,
    const torch::Tensor& X,
    const torch::Tensor& A,
    const torch::Tensor& B) {
  TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

  TORCH_CHECK(W.scalar_type() == torch::kFloat32, "W must be float32");
  TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");
  TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
  TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");

  TORCH_CHECK(W.dim() == 2, "W must be rank-2");
  TORCH_CHECK(X.dim() == 2, "X must be rank-2");
  TORCH_CHECK(A.dim() == 2, "A must be rank-2");
  TORCH_CHECK(B.dim() == 2, "B must be rank-2");

  TORCH_CHECK(W.size(0) == W.size(1), "W must be square");
  TORCH_CHECK(X.size(0) == X.size(1), "X must be square");
  TORCH_CHECK(W.size(0) == X.size(0), "W and X must share dimension d");
  TORCH_CHECK(A.size(0) == W.size(0), "A must have shape d x r");
  TORCH_CHECK(B.size(0) == W.size(0), "B must have shape d x r");
  TORCH_CHECK(A.size(1) == B.size(1), "A and B must share rank r");
}

}  // namespace

torch::Tensor forward(
    torch::Tensor W,
    torch::Tensor X,
    torch::Tensor A,
    torch::Tensor B) {
  check_inputs(W, X, A, B);
  auto bt = B.transpose(0, 1).contiguous();
  auto dense = torch::matmul(W, X);
  auto low_rank = torch::matmul(A, torch::matmul(bt, X));
  return dense + low_rank;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "LoRA forward");
}
"""


@dataclass
class OptimizeLoraResult(WorkflowResult):
    status: str
    summary: str
    submission_root: str
    promoted_source_path: str
    artifact_created: bool
    bootstrap_family: str = "baseline"
    notes: list[str] = field(default_factory=list)


class OptimizeLoraRunner:
    def __init__(self) -> None:
        self._artifact_name = "optimized_lora.cu"

    def _submission_root(self) -> Path:
        return Path.cwd()

    def _artifact_path(self) -> Path:
        return self._submission_root() / self._artifact_name

    def _write_bootstrap_artifact(self) -> Path:
        artifact_path = self._artifact_path()
        artifact_path.write_text(BASELINE_SOURCE, encoding="utf-8")
        return artifact_path

    def run(self, context: RuntimeContext) -> OptimizeLoraResult:
        now = strftime("%Y-%m-%dT%H:%M:%S")
        artifact_path = self._write_bootstrap_artifact()
        return OptimizeLoraResult(
            workflow="optimize-lora",
            started_at=context.started_at,
            finished_at=now,
            status="bootstrap_ready",
            summary=(
                "Wrote a baseline single-file LoRA extension to the submission root. "
                "The optimization loop still needs to be implemented."
            ),
            submission_root=str(self._submission_root()),
            promoted_source_path=str(artifact_path),
            artifact_created=artifact_path.exists(),
            notes=[
                "Bootstrap artifact is a correctness-first ATen-backed extension.",
                "Current workflow guarantees immediate optimized_lora.cu creation before search work.",
            ],
        )


def OptimizeLoraWorkflow() -> Workflow:
    return Workflow(
        name="optimize-lora",
        description="Bootstrap and optimize a LoRA-style CUDA extension.",
        runner=OptimizeLoraRunner().run,
    )
