#pragma once

#include <torch/extension.h>


// q,k: [seq_len, num_heads, head_dim]
void rotary_pos_encoding(torch::Tensor& positions, torch::Tensor& in);