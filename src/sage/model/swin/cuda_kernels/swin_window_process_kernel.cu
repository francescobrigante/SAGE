/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

int best_block_dim(int feat_dim){
    int best_dim;
    if (feat_dim < 384){
        best_dim = 64;
    }
    else{
        if (feat_dim < 1024){
            best_dim = 128;
        }
        else{
            best_dim = 256;
        }
    }
    return best_dim;
}


// K1: roll (cyclic shift) + window partition.
// input:  [B, H, W, C]
// output: [B*nH*nW, window_h, window_w, C]
// grid:   (window_w, window_h, B*nH*nW)  — x=cols(W-dim), y=rows(H-dim), z=flat_window
template <typename T>
__global__ void roll_and_window_partition_forward_cuda_kernel(
    T* input,
    T* output,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w,
    const int nH,
    const int nW){

    int index = threadIdx.x;
    for (int i = index; i < C; i += blockDim.x) {
        // flat output index: (flat_win * window_h + ly) * window_w * C + lx * C + i
        int offset = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * C + i;
        // blockIdx.y = ly in [0, window_h-1], blockIdx.x = lx in [0, window_w-1]
        // blockIdx.z = flat_win = b*nH*nW + wrow*nW + wcol
        int input_offset = blockIdx.z / (nH * nW) * H * W * C +
            (blockIdx.z % (nH * nW) / nW * window_h + blockIdx.y - shift_h + H) % H * W * C +
            (blockIdx.z % nW * window_w + blockIdx.x - shift_w + W) % W * C +
            i;
        output[offset] = (T)(__ldg(input + input_offset));
    }
}


// K2: backward of K1 — scatter gradients back from window layout to spatial layout.
// grad_in:  [B*nH*nW, window_h, window_w, C]
// grad_out: [B, H, W, C]
// grid:     (W, H, B)
template <typename T>
__global__ void roll_and_window_partition_backward_cuda_kernel(
    T* grad_in,
    T* grad_out,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w,
    const int nH,
    const int nW){

    int index = threadIdx.x;
    for (int i = index; i < C; i += blockDim.x) {
        // output offset in [B, H, W, C]
        int offset = ((blockIdx.z * H + blockIdx.y) * W + blockIdx.x) * C + i;
        // blockIdx.y=y in [0,H-1], blockIdx.x=x in [0,W-1], blockIdx.z=b
        int input_offset =
            (blockIdx.z * nH * nW +
             (blockIdx.y + shift_h + H) % H / window_h * nW +
             (blockIdx.x + shift_w + W) % W / window_w) * window_h * window_w * C +
            (blockIdx.y + shift_h + H) % H % window_h * window_w * C +
            (blockIdx.x + shift_w + W) % W % window_w * C +
            i;
        grad_out[offset] = (T)(__ldg(grad_in + input_offset));
    }
}


// K3: window merge + reverse roll (inverse of K1).
// input:  [B*nH*nW, window_h, window_w, C]
// output: [B, H, W, C]
// grid:   (W, H, B)
//
// Bug fixes vs. original:
//   - '* nH' on the window-row term corrected to '* nW' (row-major: nW cols per row)
//   - Added explicit '% H' and '% W' before the intra-window modulo
template <typename T>
__global__ void window_merge_and_roll_forward_cuda_kernel(
    T* input,
    T* output,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w,
    const int nH,
    const int nW){

    int index = threadIdx.x;
    for (int i = index; i < C; i += blockDim.x) {
        // output offset in [B, H, W, C]
        int offset = ((blockIdx.z * H + blockIdx.y) * W + blockIdx.x) * C + i;
        // blockIdx.y=y in [0,H-1], blockIdx.x=x in [0,W-1], blockIdx.z=b
        // undo the forward roll: source position in the pre-roll window layout
        int fy = (blockIdx.y - shift_h + H) % H;
        int fx = (blockIdx.x - shift_w + W) % W;
        int input_offset =
            (blockIdx.z * nH * nW + fy / window_h * nW + fx / window_w) * window_h * window_w * C +
            (fy % window_h) * window_w * C +
            (fx % window_w) * C +
            i;
        output[offset] = (T)(__ldg(input + input_offset));
    }
}


// K4: backward of K3 — symmetric to K1 but with +shift instead of -shift.
// grad_in:  [B, H, W, C]
// grad_out: [B*nH*nW, window_h, window_w, C]
// grid:     (window_w, window_h, B*nH*nW)
template <typename T>
__global__ void window_merge_and_roll_backward_cuda_kernel(
    T* grad_in,
    T* grad_out,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w,
    const int nH,
    const int nW){

    int index = threadIdx.x;
    for (int i = index; i < C; i += blockDim.x) {
        // flat output index in [B*nH*nW, window_h, window_w, C]
        int offset = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * C + i;
        // blockIdx.y = ly in [0, window_h-1], blockIdx.x = lx in [0, window_w-1]
        // blockIdx.z = flat_win = b*nH*nW + wrow*nW + wcol
        int input_offset =
            (blockIdx.z / (nH * nW)) * H * W * C +
            (blockIdx.z % (nH * nW) / nW * window_h + blockIdx.y + shift_h + H) % H * W * C +
            (blockIdx.z % nW * window_w + blockIdx.x + shift_w + W) % W * C +
            i;
        grad_out[offset] = (T)(__ldg(grad_in + input_offset));
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Launcher functions
// ─────────────────────────────────────────────────────────────────────────────

// input: [B, H, W, C]  →  output: [B*nH*nW, window_h, window_w, C]
at::Tensor roll_and_window_partition_forward_cuda(
    at::Tensor & input,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w){

    int nH = H / window_h;
    int nW = W / window_w;

    dim3 grid(window_w, window_h, B * nH * nW);
    int blocknum = best_block_dim(C);
    dim3 block(blocknum);

    auto opts = input.options();
    at::Tensor output = at::empty({B*nH*nW, window_h, window_w, C}, opts);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "roll_and_window_partition_forward_cuda_kernel", ([&] {
        roll_and_window_partition_forward_cuda_kernel<scalar_t><<<grid, block, 0>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, H, W, C,
            shift_h, shift_w, window_h, window_w,
            nH, nW);
    }));
    return output;
}


// grad_in: [B*nH*nW, window_h, window_w, C]  →  grad_out: [B, H, W, C]
at::Tensor roll_and_window_partition_backward_cuda(
    at::Tensor & grad_in,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w){

    int nH = H / window_h;
    int nW = W / window_w;

    dim3 grid(W, H, B);
    int blocknum = best_block_dim(C);
    dim3 block(blocknum);

    auto opts = grad_in.options();
    at::Tensor grad_out = at::empty({B, H, W, C}, opts);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_in.scalar_type(), "roll_and_window_partition_backward_cuda_kernel", ([&] {
        roll_and_window_partition_backward_cuda_kernel<scalar_t><<<grid, block, 0>>>(
            grad_in.data_ptr<scalar_t>(),
            grad_out.data_ptr<scalar_t>(),
            B, H, W, C,
            shift_h, shift_w, window_h, window_w,
            nH, nW);
    }));
    return grad_out;
}


// input: [B*nH*nW, window_h, window_w, C]  →  output: [B, H, W, C]
at::Tensor window_merge_and_roll_forward_cuda(
    at::Tensor & input,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w){

    int nH = H / window_h;
    int nW = W / window_w;

    dim3 grid(W, H, B);
    int blocknum = best_block_dim(C);
    dim3 block(blocknum);

    auto opts = input.options();
    at::Tensor output = at::empty({B, H, W, C}, opts);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "window_merge_and_roll_forward_cuda_kernel", ([&] {
        window_merge_and_roll_forward_cuda_kernel<scalar_t><<<grid, block, 0>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, H, W, C,
            shift_h, shift_w, window_h, window_w,
            nH, nW);
    }));
    return output;
}


// grad_in: [B, H, W, C]  →  grad_out: [B*nH*nW, window_h, window_w, C]
at::Tensor window_merge_and_roll_backward_cuda(
    at::Tensor & grad_in,
    const int B,
    const int H,
    const int W,
    const int C,
    const int shift_h,
    const int shift_w,
    const int window_h,
    const int window_w){

    int nH = H / window_h;
    int nW = W / window_w;

    dim3 grid(window_w, window_h, B * nH * nW);
    int blocknum = best_block_dim(C);
    dim3 block(blocknum);

    auto opts = grad_in.options();
    at::Tensor grad_out = at::empty({B*nH*nW, window_h, window_w, C}, opts);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_in.scalar_type(), "window_merge_and_roll_backward_cuda_kernel", ([&] {
        window_merge_and_roll_backward_cuda_kernel<scalar_t><<<grid, block, 0>>>(
            grad_in.data_ptr<scalar_t>(),
            grad_out.data_ptr<scalar_t>(),
            B, H, W, C,
            shift_h, shift_w, window_h, window_w,
            nH, nW);
    }));
    return grad_out;
}
