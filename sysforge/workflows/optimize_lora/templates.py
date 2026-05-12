from __future__ import annotations

import re

COMMON_INCLUDES = """#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
"""

COMMON_PREFIX = """
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
"""

COMMON_SUFFIX = """
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "LoRA forward");
}
"""

FORWARD_BODY_RE = re.compile(
    r"torch::Tensor forward\(\s*torch::Tensor W,\s*torch::Tensor X,\s*torch::Tensor A,\s*torch::Tensor B\)\s*\{\s*check_inputs\(W, X, A, B\);\s*(?P<body>.*?)\s*\}\s*PYBIND11_MODULE",
    re.DOTALL,
)
INCLUDE_RE = re.compile(r"^\s*#include.*$", re.MULTILINE)


def _render_single_forward(function_name: str, body: str) -> str:
    return f"""
torch::Tensor {function_name}(
    torch::Tensor W,
    torch::Tensor X,
    torch::Tensor A,
    torch::Tensor B) {{
  check_inputs(W, X, A, B);
{body}
}}
"""


def render_source_from_body(body: str) -> str:
    return COMMON_INCLUDES + COMMON_PREFIX + _render_single_forward("forward", body.rstrip()) + COMMON_SUFFIX


def extract_forward_body(source: str) -> str:
    match = FORWARD_BODY_RE.search(source)
    if match:
        return match.group("body").rstrip()
    return source.strip()


def sanitize_generated_body(generated: str) -> str:
    body = extract_forward_body(generated)
    body = INCLUDE_RE.sub("", body).strip()
    if "PYBIND11_MODULE" in body:
        body = body.split("PYBIND11_MODULE", 1)[0].strip()
    if "torch::Tensor forward" in body:
        body = extract_forward_body(body)
    return body.rstrip()


BASELINE_SOURCE = render_source_from_body(
    """  auto bt = B.transpose(0, 1).contiguous();
  auto dense = torch::matmul(W, X);
  auto low_rank = torch::matmul(A, torch::matmul(bt, X));
  return dense + low_rank;"""
)
