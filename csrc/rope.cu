#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ops.h>

namespace rope{

// a: (seq_len, head_dim)
__global__ void rotary_pos_encoding_seq(const float* a, const int seq_len, const int head_dim){
    return;
}

// a: (head_dim)
__global__ void rotary_pos_encoding(const float* a, const float theta, const int m, const int head_dim){
    return;
}


}

void rope_rotary_pos_encoding(torch::Tensor& positions, torch::Tensor& in){

}