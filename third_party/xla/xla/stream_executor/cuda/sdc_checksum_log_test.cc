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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace se = stream_executor;

namespace {

class SdcChecksumLogTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(platform_,
                            se::PlatformManager::PlatformWithName("CUDA"));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
  }

  se::Platform* platform_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
};

TEST_F(SdcChecksumLogTest, CreateSdcLogOnDevice_AllocatesEmptyLog) {
  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceMemory<se::cuda::SdcChecksumLogHeader> device_log,
      se::cuda::CreateSdcLogOnDevice(*stream_, /*max_entries=*/10));
  TF_ASSERT_OK_AND_ASSIGN(auto host_log,
                          se::cuda::ReadSdcLogFromDevice(*stream_, device_log));

  EXPECT_EQ(host_log.size(), 0);
}

TEST_F(SdcChecksumLogTest, CreateSdcLogOnDevice_AllocatesEnoughSpace) {
  constexpr uint32_t kMaxEntries = 10;
  constexpr size_t kExpectedSizeBytes =
      sizeof(se::cuda::SdcChecksumLogHeader) +
      kMaxEntries * sizeof(se::cuda::SdcChecksumLogEntry);

  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceMemory<se::cuda::SdcChecksumLogHeader> device_log,
      se::cuda::CreateSdcLogOnDevice(*stream_, kMaxEntries));

  EXPECT_EQ(device_log.size(), kExpectedSizeBytes);
}

TEST_F(SdcChecksumLogTest, CreateSdcLogOnDevice_InitializesHeader) {
  constexpr uint32_t kMaxEntries = 10;

  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceMemory<se::cuda::SdcChecksumLogHeader> device_log,
      se::cuda::CreateSdcLogOnDevice(*stream_, kMaxEntries));

  se::cuda::SdcChecksumLogHeader header;
  TF_ASSERT_OK(stream_->Memcpy(&header, device_log, sizeof(header)));
  TF_ASSERT_OK(stream_->BlockHostUntilDone());
  EXPECT_EQ(header.write_idx, 0);
  EXPECT_EQ(header.capacity, kMaxEntries);
}

}  // namespace
