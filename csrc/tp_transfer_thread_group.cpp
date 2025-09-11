/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tp_transfer_thread_group.h"
#include "transfer.cuh"
#include <stdexcept>

namespace flexkv {

TPTransferThreadGroup::TPTransferThreadGroup(
    int num_gpus, const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
    torch::Tensor &cpu_blocks, int dp_group_id,
    torch::Tensor &gpu_kv_strides_tensor,
    torch::Tensor &gpu_block_strides_tensor,
    torch::Tensor &gpu_chunk_sizes_tensor) {

  num_gpus_ = num_gpus;
  
  gpu_kv_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_block_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_chunk_sizes_in_bytes_ = new int64_t[num_gpus];
  
  int64_t* kv_strides_ptr = gpu_kv_strides_tensor.data_ptr<int64_t>();
  int64_t* block_strides_ptr = gpu_block_strides_tensor.data_ptr<int64_t>();
  int64_t* chunk_sizes_ptr = gpu_chunk_sizes_tensor.data_ptr<int64_t>();
  
  for (int i = 0; i < num_gpus; i++) {
    gpu_kv_strides_in_bytes_[i] = kv_strides_ptr[i];
    gpu_block_strides_in_bytes_[i] = block_strides_ptr[i];
    gpu_chunk_sizes_in_bytes_[i] = chunk_sizes_ptr[i];
  }

  queues_.resize(num_gpus_);
  mtxs_   = std::vector<std::mutex>(num_gpus_);
  cvs_    = std::vector<std::condition_variable>(num_gpus_);

  int num_layers = gpu_blocks[0].size();
  cudaMallocHost((void **)&gpu_blocks_,
                 num_gpus_ * num_layers * sizeof(void *));
  for (int i = 0; i < num_gpus_; ++i) {
    for (int j = 0; j < num_layers; ++j) {
      gpu_blocks_[i * num_layers + j] = gpu_blocks[i][j].data_ptr();
    }
  }

  cpu_blocks_ = cpu_blocks.data_ptr();

  dp_group_id_ = dp_group_id;
  streams_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; i += 1) {
    cudaSetDevice(dp_group_id * num_gpus_ + i);
    cudaStreamCreate(&streams_[i]);
  }
  // create the thread pool
  stop_pool_=false;
  for (int i = 0; i < num_gpus_; ++i) {
    threads_.emplace_back([this, i]() {
      int device_id = dp_group_id_ * num_gpus_ + i;
      cudaSetDevice(device_id);  // only once

      while (true) {
        Task task;
        {
          std::unique_lock<std::mutex> lk(mtxs_[i]);
          cvs_[i].wait(lk, [&]{ return stop_pool_ || !queues_[i].empty(); });
          if (stop_pool_ && queues_[i].empty()) return;

          task = std::move(queues_[i].front());
          queues_[i].pop();
        }
        task();  // 
      }
    });
  }

}

TPTransferThreadGroup::~TPTransferThreadGroup() {
  stop_pool_ = true;
  for (auto& cv : cvs_) cv.notify_all();
  for (auto& t : threads_) if (t.joinable()) t.join();

  cudaFreeHost(gpu_blocks_);
  
  delete[] gpu_kv_strides_in_bytes_;
  delete[] gpu_block_strides_in_bytes_;
  delete[] gpu_chunk_sizes_in_bytes_;
}

std::future<void> TPTransferThreadGroup::enqueue_for_gpu(int gpu_idx, Task task) {
  auto pkg = std::make_shared<std::packaged_task<void()>>(std::move(task));
  auto fut = pkg->get_future();
  {
      std::lock_guard<std::mutex> lk(mtxs_[gpu_idx]);
      queues_[gpu_idx].emplace([pkg]{ (*pkg)(); });
  }
  cvs_[gpu_idx].notify_one();
  return fut;
}

void TPTransferThreadGroup::tp_group_transfer(
    const torch::Tensor &gpu_block_id_tensor,
    const torch::Tensor &cpu_block_id_tensor,
    const int64_t cpu_kv_stride_in_bytes,
    const int64_t cpu_layer_stride_in_bytes,
    const int64_t cpu_block_stride_in_bytes,
    const int64_t cpu_chunk_size_in_bytes, const int transfer_sms,
    const bool is_host_to_device, const bool use_ce_transfer,
    const int layer_id, const int layer_granularity, const bool is_mla) {

  std::atomic<bool> failed{false};
  std::string error_msg;
  // threads_.clear();
  // threads_.reserve(num_gpus_);

  // Barrier sync_point(num_gpus_);
  std::vector<std::future<void>> futures;
  futures.reserve(num_gpus_);

  for (int i=0; i<num_gpus_; ++i){
    futures.emplace_back(enqueue_for_gpu(i, [&, i]() {
      try {
        int num_blocks = gpu_block_id_tensor.numel();
        int num_layers = layer_granularity;

        int64_t *gpu_block_ids =
            static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
        int64_t *cpu_block_ids =
            static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
        void **gpu_layer_ptrs =
            static_cast<void **>(gpu_blocks_ + i * num_layers + layer_id);
        void *cpu_ptr = cpu_blocks_;
        int64_t cpu_startoff_inside_chunks =
            is_mla ? 0 : i * gpu_chunk_sizes_in_bytes_[i];
      
        flexkv::transfer_kv_blocks(
          num_blocks, layer_id, layer_granularity, gpu_block_ids,
          gpu_layer_ptrs, gpu_kv_strides_in_bytes_[i], gpu_block_strides_in_bytes_[i],
          cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
          cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
          cpu_startoff_inside_chunks, gpu_chunk_sizes_in_bytes_[i], streams_[i],
          transfer_sms, is_host_to_device, use_ce_transfer, is_mla
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          failed = true;
          error_msg = cudaGetErrorString(err);
        }
      } catch (const std::exception &e) {
        failed = true;
        error_msg = e.what();
      }

    }));
  }

  for (auto &f : futures){
    f.get();
  }

  if (failed) {
    throw std::runtime_error("tp_group_transfer failed: " + error_msg);
  }
}

} // namespace flexkv
