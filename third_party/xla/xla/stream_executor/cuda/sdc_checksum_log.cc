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

#include "xla/stream_executor/cuda/sdc_checksum_log.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor::cuda {

absl::StatusOr<DeviceMemory<SdcChecksumLogHeader>> CreateSdcLogOnDevice(
    Stream& stream, uint32_t max_entries) {
  DeviceMemory<uint8_t> data = stream.parent()->AllocateArray<uint8_t>(
      sizeof(SdcChecksumLogHeader) + max_entries * sizeof(SdcChecksumLogEntry));
  if (data.is_null()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate SdcChecksumLog with %u entries", max_entries));
  }

  const SdcChecksumLogHeader empty_header{
      /*write_idx=*/0,
      /*capacity=*/max_entries,
  };
  TF_RETURN_IF_ERROR(stream.Memcpy(&data, &empty_header, sizeof(empty_header)));
  return DeviceMemory<SdcChecksumLogHeader>(data);
}

absl::StatusOr<std::vector<SdcChecksumLogEntry>> ReadSdcLogFromDevice(
    Stream& stream, const DeviceMemory<SdcChecksumLogHeader>& device_log) {
  SdcChecksumLogHeader header;
  TF_RETURN_IF_ERROR(stream.Memcpy(&header, device_log, sizeof(header)));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());

  const size_t initialized_entries =
      std::min(header.capacity, header.write_idx);
  const size_t initialized_bytes =
      initialized_entries * sizeof(SdcChecksumLogEntry);
  std::vector<SdcChecksumLogEntry> entries(initialized_entries);
  TF_RETURN_IF_ERROR(
      stream.Memcpy(entries.data(),
                    device_log.GetByteSlice(sizeof(header), initialized_bytes),
                    initialized_bytes));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());

  return entries;
}

}  // namespace stream_executor::cuda
