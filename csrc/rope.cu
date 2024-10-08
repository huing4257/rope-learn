#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace rope{

__global__ void rotary_pos_encoding(const float* a, const float theta){
    return;
}
}