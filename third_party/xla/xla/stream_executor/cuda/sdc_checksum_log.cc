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

#include <cstdint>
#include <cstdlib>
#include <memory>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor::cuda {

void SdcChecksumLog::Delete(void* ptr) { free(ptr); }

std::unique_ptr<SdcChecksumLog, decltype(&SdcChecksumLog::Delete)>
SdcChecksumLog::Create(uint32_t max_entries) {
  uint8_t* data =
      reinterpret_cast<uint8_t*>(malloc(MaxSizeForEntries(max_entries)));
  CHECK(data != nullptr) << "Failed to allocate SdcChecksumLog";
  return std::unique_ptr<SdcChecksumLog, decltype(&SdcChecksumLog::Delete)>(
      new (data) SdcChecksumLog{
          /*write_idx=*/0,
          /*capacity=*/max_entries,
      },
      SdcChecksumLog::Delete);
}

absl::StatusOr<DeviceMemory<SdcChecksumLog>> SdcChecksumLog::CreateOnDevice(
    Stream& stream, uint32_t max_entries) {
  DeviceMemory<uint8_t> data =
      stream.parent()->AllocateArray<uint8_t>(MaxSizeForEntries(max_entries));
  if (data.is_null()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate SdcChecksumLog with %u entries", max_entries));
  }

  const SdcChecksumLog empty_header{
      /*write_idx=*/0,
      /*capacity=*/max_entries,
  };
  TF_RETURN_IF_ERROR(stream.Memcpy(&data, &empty_header, sizeof(empty_header)));
  return DeviceMemory<SdcChecksumLog>(data);
}

absl::StatusOr<
    std::unique_ptr<SdcChecksumLog, decltype(&SdcChecksumLog::Delete)>>
SdcChecksumLog::CreateFromDeviceMemory(
    Stream& stream, DeviceMemory<SdcChecksumLog>& device_log) {
  SdcChecksumLog header;
  TF_RETURN_IF_ERROR(stream.Memcpy(&header, device_log, sizeof(header)));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());

  auto host_log = Create(header.capacity);
  TF_RETURN_IF_ERROR(
      stream.Memcpy(host_log.get(), device_log, header.UsedSizeBytes()));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return host_log;
}

absl::StatusOr<DeviceMemory<SdcChecksumLog>> SdcChecksumLog::ToDeviceMemory(
    Stream& stream) {
  DeviceMemory<uint8_t> data =
      stream.parent()->AllocateArray<uint8_t>(MaxSizeBytes());
  if (data.is_null()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to allocate %u bytes for SdcChecksumLog", MaxSizeBytes()));
  }

  TF_RETURN_IF_ERROR(stream.Memcpy(&data, this, UsedSizeBytes()));
  return DeviceMemory<SdcChecksumLog>(data);
}

}  // namespace stream_executor::cuda
