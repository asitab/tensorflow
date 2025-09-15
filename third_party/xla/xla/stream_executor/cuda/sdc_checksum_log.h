/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_STREAM_EXECUTOR_CUDA_SDC_CHECKSUM_LOG_H_
#define XLA_STREAM_EXECUTOR_CUDA_SDC_CHECKSUM_LOG_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor::cuda {

struct SdcChecksumLogEntry {
  // An ID that uniquely identifies a thunk and its specific input or output
  // buffer.
  uint32_t entry_id;
  uint32_t checksum;
};

// A log of checksums calculated on the device, that can be used to detect
// nondeterminism or silent data corruption.
//
// The log is intended to be write-only on the device and read-only on host.
// It is implemented with a flexible array member to enable moving it from
// device to host in a single memcpy.
struct SdcChecksumLog {
 private:
  // Use provided factory methods to create instances.
  SdcChecksumLog() : SdcChecksumLog(/*write_idx=*/0, /*capacity=*/0) {}
  explicit SdcChecksumLog(uint32_t write_idx, uint32_t capacity)
      : write_idx(write_idx), capacity(capacity) {}

  static void Delete(void* ptr);

 public:
  // The first entry in `entries` that has not been written to. May be bigger
  // than `capacity` if the log was truncated.
  uint32_t write_idx;
  // The number of `entries` the log can hold.
  uint32_t capacity;
  SdcChecksumLogEntry entries[];

  // Use factory methods instead.
  SdcChecksumLog(const SdcChecksumLog&) = delete;
  SdcChecksumLog& operator=(const SdcChecksumLog&) = delete;
  SdcChecksumLog(SdcChecksumLog&&) = delete;
  SdcChecksumLog& operator=(SdcChecksumLog&&) = delete;

  // Creates a new SdcChecksumLog in host memory with enough capacity to hold
  // max_entries of SdcLogEntries.
  static std::unique_ptr<SdcChecksumLog, decltype(&SdcChecksumLog::Delete)>
  Create(uint32_t max_entries);

  // Allocates an empty SdcChecksumLog on the device with enough capacity to
  // hold max_entries of SdcLogEntries.
  static absl::StatusOr<DeviceMemory<SdcChecksumLog>> CreateOnDevice(
      Stream& stream, uint32_t max_entries);

  // Retrieves the log from device memory.
  static absl::StatusOr<
      std::unique_ptr<SdcChecksumLog, decltype(&SdcChecksumLog::Delete)>>
  CreateFromDeviceMemory(Stream& stream, DeviceMemory<SdcChecksumLog>& log);

  // Copies the log to device memory.
  absl::StatusOr<DeviceMemory<SdcChecksumLog>> ToDeviceMemory(Stream& stream);

  // Number of valid entries that can be read from the log.
  size_t Size() const { return std::min(write_idx, capacity); }

  // Number of bytes used by the log and all initialized entries.
  size_t UsedSizeBytes() const {
    return sizeof(SdcChecksumLog) + sizeof(SdcChecksumLogEntry) * Size();
  }

  // Number of bytes needed to store the log and all entries, including
  // uninitialized ones.
  size_t MaxSizeBytes() const { return MaxSizeForEntries(capacity); }

  // Number of bytes needed to store a log with the given capacity.
  static size_t MaxSizeForEntries(uint32_t capacity) {
    return sizeof(SdcChecksumLog) + sizeof(SdcChecksumLogEntry) * capacity;
  }
};

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_SDC_CHECKSUM_LOG_H_
