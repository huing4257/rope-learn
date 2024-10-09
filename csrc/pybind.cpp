#include <torch/extension.h>

#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(rope, m) {
  m.def("rotary_pos_encoding(Tensor input) -> ()");
}

TORCH_LIBRARY_IMPL(rope, CUDA, m) {
  m.impl("rotary_pos_encoding", rotary_pos_encoding);
}


