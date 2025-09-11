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

#include <cstdint>
#include <vector>

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

struct SdcChecksumLogHeader {
  // The first entry in `entries` that has not been written to. May be bigger
  // than `capacity` if the log was truncated.
  uint32_t write_idx;
  // The number of `entries` the log can hold.
  uint32_t capacity;
};

// Allocates an empty SdcChecksumLog on the device with enough capacity to
// hold max_entries of SdcLogEntries.
//
// Contents of the log can be retrieved with ReadSdcLogFromDevice.
absl::StatusOr<DeviceMemory<SdcChecksumLogHeader>> CreateSdcLogOnDevice(
    Stream& stream, uint32_t max_entries);

// Reads all entries from the device log into host memory.
//
// Returned vector contains all initialized entries. If the log overflowed,
// excess elements are silently discarded.
absl::StatusOr<std::vector<SdcChecksumLogEntry>> ReadSdcLogFromDevice(
    Stream& stream, const DeviceMemory<SdcChecksumLogHeader>& log);

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_SDC_CHECKSUM_LOG_H_
