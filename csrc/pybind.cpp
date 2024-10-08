#include <torch/extension.h>

#include "ops.h"

TORCH_LIBRARY(rope, m) {
  m.def("rotary_pos_encoding("
        "Tensor positions, "
        "Tensor in) -> ()");
}

TORCH_LIBRARY_IMPL(lingohpcops, CUDA, m) {
  m.impl("rotary_embedding", rotary_pos_encoding);
}


