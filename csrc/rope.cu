#include <cuda.h>
#include <cuda_runtime.h>
#include <ops.h>
#include <torch/extension.h>
#include <torch_utils.h>

namespace rope {

    // a: (seq_len, head_dim)
    __global__ void rotary_pos_encoding_seq(float *__restrict__ a, 
                                            const int seq_len,
                                            const int head_dim,
                                            const float log_2_theta_base) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x < head_dim / 2 && y < seq_len) {
            float x_1 = a[y * head_dim + x * 2];
            float x_2 = a[y * head_dim + x * 2 + 1];
            // float sin, cos;
            // float theta = y * exp2f(-log_2_theta_base * (float)x * 2 / head_dim);
            // sincosf(theta, &sin, &cos);
            // a[y * head_dim + x * 2] = x_1 * cos - x_2 * sin;
            // a[y * head_dim + x * 2 + 1] = x_1 * sin + x_2 * cos;
            a[y * head_dim + x * 2] = x_1 + y;
            a[y * head_dim + x * 2 + 1] = x_2 + y;
        }
        return;
    }

    // a: (head_dim)
    __global__ void rotary_pos_encoding_with_pos(float *__restrict__ a,
                                                 const int *__restrict__ m, 
                                                 const int seq_len, 
                                                 const int head_dim, 
                                                 const float log_2_theta_base) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x < head_dim / 2 && y < seq_len) {
            float x_1 = a[x * 2];
            float x_2 = a[x * 2 + 1];
            float sin, cos;
            float theta = m[y] * exp2f(-log_2_theta_base * (float)x * 2 / head_dim);
            sincosf(theta, &sin, &cos);
            a[x * 2] = x_1 * cos - x_2 * sin;
            a[x * 2 + 1] = x_1 * sin + x_2 * cos;
        }
        return;
    }

} // namespace rope

// input: (seq_len, head_dim)
void rotary_pos_encoding(torch::Tensor &input) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dim() == 2, "rotary_pos_encoding: input must be 2D");
    auto a = input.mutable_data_ptr<float>();
    auto seq_len = input.size(0);
    auto head_dim = input.size(1);
    auto log_2_theta_base = log2(10000.0);
    dim3 threads(32, 32);
    dim3 blocks((head_dim / 2 + threads.x - 1) / threads.x, (seq_len + threads.y - 1) / threads.y);
    rope::rotary_pos_encoding_seq<<<blocks, threads>>>(a, seq_len, head_dim, log_2_theta_base);
}

// positions: (seq_len), input: (seq_len, head_dim)
void rotary_pos_encoding_with_pos(torch::Tensor &positions, torch::Tensor &input) {
    CHECK_INPUT(positions);
    CHECK_INPUT(input);
    TORCH_CHECK(positions.dim() == 1, "rotary_pos_encoding_with_pos: positions must be 1D");
    TORCH_CHECK(input.dim() == 2, "rotary_pos_encoding_with_pos: input must be 2D");
    TORCH_CHECK(positions.size(0) == input.size(0), "rotary_pos_encoding_with_pos: positions and input must have the same length");
    auto a = input.mutable_data_ptr<float>();
    auto m = positions.data_ptr<int>();
    auto seq_len = input.size(0);
    auto head_dim = input.size(1);
    auto log_2_theta_base = log2(10000.0);
    dim3 threads(32, 16);
    dim3 blocks((head_dim / 2 + threads.x - 1) / threads.x, (positions.size(0) + threads.y - 1) / threads.y);
    rope::rotary_pos_encoding_with_pos<<<blocks, threads>>>(a, m, seq_len, head_dim, log_2_theta_base);
}