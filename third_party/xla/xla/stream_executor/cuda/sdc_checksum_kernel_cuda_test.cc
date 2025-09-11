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

#include "xla/stream_executor/cuda/sdc_checksum_kernel_cuda.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/sdc_checksum_log.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"  // IWYU pragma: keep, required for KernelType::FactoryType::Create
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace se = stream_executor;

namespace {

class ChecksumKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(platform_,
                            se::PlatformManager::PlatformWithName("CUDA"));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
  }

  template <typename T>
  absl::StatusOr<se::DeviceMemory<T>> CheckNull(
      se::DeviceMemory<T> device_memory, absl::string_view name) {
    if (device_memory.is_null()) {
      return absl::InternalError(
          absl::StrFormat("Device memory for %s is null", name));
    }
    return device_memory;
  }

  template <typename T>
  absl::Status AppendChecksumOnDevice(
      uint32_t entry_id, const T& input,
      se::DeviceMemory<se::cuda::SdcChecksumLogHeader> device_checksum_log,
      stream_executor::ThreadDim dim = stream_executor::ThreadDim(1, 1, 1),
      std::optional<size_t> scratch_size_elements = std::nullopt) {
    // Load kernel
    TF_ASSIGN_OR_RETURN(se::KernelLoaderSpec spec,
                        se::cuda::GetSdcChecksumKernelSpec());
    TF_ASSIGN_OR_RETURN(
        auto kernel,
        se::cuda::SdcChecksumKernel::KernelType::FactoryType::Create(executor_,
                                                                     spec));

    // Setup device buffers
    TF_ASSIGN_OR_RETURN(se::DeviceMemory<uint8_t> device_input,
                        CheckNull(executor_->AllocateArray<uint8_t>(
                                      input.size() * sizeof(input[0])),
                                  "input"));
    auto cleanup_input =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_input); });

    // Call kernel
    TF_RETURN_IF_ERROR(
        stream_->Memcpy(&device_input, input.data(), input.size()));

    TF_RETURN_IF_ERROR(kernel.Launch(
        dim, stream_executor::BlockDim(1, 1, 1), stream_.get(), entry_id,
        device_input, device_input.ElementCount(), device_checksum_log));

    TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());

    // The result gets stored in device_checksum_log.
    return absl::OkStatus();
  }

  se::Platform* platform_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
};

template <size_t N>
struct TestInput {
  std::array<uint8_t, N> input;
  uint32_t expected_checksum;
};

template <size_t N>
TestInput<N> GetTestInput() {
  std::array<uint8_t, N> input;
  absl::BitGen gen;
  std::generate_n(input.begin(), N,
                  [&]() { return absl::Uniform<uint8_t>(gen); });

  uint32_t checksum = 0;
  for (size_t i = 0; i < N; i += sizeof(uint32_t)) {
    uint32_t chunk = 0;
    memcpy(&chunk, &input[i], std::min(sizeof(chunk), N - i));
    checksum ^= chunk;
  }

  return TestInput{
      input,
      checksum,
  };
}

const auto [kInput32k, kExpectedChecksum32k] = GetTestInput<32 * 1024>();
const auto [kInput64k, kExpectedChecksum64k] = GetTestInput<64 * 1024>();
const auto [kInput128k, kExpectedChecksum128k] = GetTestInput<128 * 1024>();
constexpr uint32_t kEntryId1 = 123;
constexpr uint32_t kEntryId2 = 456;
constexpr uint32_t kEntryId3 = 789;
constexpr se::ThreadDim kTestDim(2, 4, 8);

TEST_F(ChecksumKernelTest, ComputesCorrectChecksum_OneThread_MultipleOf32Bit) {
  const auto [kInput, kExpectedChecksum] = GetTestInput<1024>();
  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceMemory<se::cuda::SdcChecksumLogHeader> device_log,
      se::cuda::CreateSdcLogOnDevice(*stream_, /*max_entries=*/10));
  auto cleanup_log =
      absl::MakeCleanup([&]() { executor_->Deallocate(&device_log); });

  EXPECT_OK(AppendChecksumOnDevice(/*entry_id=*/0, kInput, device_log));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log,
                          se::cuda::ReadSdcLogFromDevice(*stream_, device_log));
  ASSERT_GE(host_log.size(), 1);
  EXPECT_EQ(host_log[0].checksum, kExpectedChecksum);
}

TEST_F(ChecksumKernelTest,
       ComputesCorrectChecksum_OneThread_NonMultipleOf32Bit) {
  const auto [kInput, kExpectedChecksum] = GetTestInput<1023>();
  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceMemory<se::cuda::SdcChecksumLogHeader> device_log,
      se::cuda::CreateSdcLogOnDevice(*stream_, /*max_entries=*/10));
  auto cleanup_log =
      absl::MakeCleanup([&]() { executor_->Deallocate(&device_log); });

  EXPECT_OK(AppendChecksumOnDevice(/*entry_id=*/0, kInput, device_log));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log,
                          se::cuda::ReadSdcLogFromDevice(*stream_, device_log));
  ASSERT_GE(host_log.size(), 1);
  EXPECT_EQ(host_log[0].checksum, kExpectedChecksum);
}

TEST_F(ChecksumKernelTest, ComputesCorrectChecksumParallel64k) {
  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceMemory<se::cuda::SdcChecksumLogHeader> device_log,
      se::cuda::CreateSdcLogOnDevice(*stream_, /*max_entries=*/10));
  auto cleanup_log =
      absl::MakeCleanup([&]() { executor_->Deallocate(&device_log); });

  EXPECT_OK(AppendChecksumOnDevice(/*entry_id=*/0, kInput64k, device_log,
                                   se::ThreadDim(2, 4, 8)));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log,
                          se::cuda::ReadSdcLogFromDevice(*stream_, device_log));
  ASSERT_GE(host_log.size(), 1);
  EXPECT_EQ(host_log[0].checksum, kExpectedChecksum64k);
}

TEST_F(ChecksumKernelTest, ComputesCorrectChecksumWithLimitedScratchSpace) {
  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceMemory<se::cuda::SdcChecksumLogHeader> device_log,
      se::cuda::CreateSdcLogOnDevice(*stream_, /*max_entries=*/10));
  auto cleanup_log =
      absl::MakeCleanup([&]() { executor_->Deallocate(&device_log); });

  EXPECT_OK(AppendChecksumOnDevice(/*entry_id=*/0, kInput64k, device_log,
                                   se::ThreadDim(2, 4, 8),
                                   /*scratch_size_elements=*/16));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log,
                          se::cuda::ReadSdcLogFromDevice(*stream_, device_log));
  ASSERT_GE(host_log.size(), 1);
  EXPECT_EQ(host_log[0].checksum, kExpectedChecksum64k);
}

TEST_F(ChecksumKernelTest, AppendsChecksumsToLog) {
  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceMemory<se::cuda::SdcChecksumLogHeader> device_log,
      se::cuda::CreateSdcLogOnDevice(*stream_, /*max_entries=*/10));
  auto cleanup_log =
      absl::MakeCleanup([&]() { executor_->Deallocate(&device_log); });

  EXPECT_OK(AppendChecksumOnDevice(kEntryId1, kInput32k, device_log, kTestDim));
  EXPECT_OK(AppendChecksumOnDevice(kEntryId2, kInput64k, device_log, kTestDim));
  EXPECT_OK(
      AppendChecksumOnDevice(kEntryId3, kInput128k, device_log, kTestDim));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log,
                          se::cuda::ReadSdcLogFromDevice(*stream_, device_log));
  ASSERT_GE(host_log.size(), 3);
  EXPECT_EQ(host_log[0].entry_id, kEntryId1);
  EXPECT_EQ(host_log[0].checksum, kExpectedChecksum32k);
  EXPECT_EQ(host_log[1].entry_id, kEntryId2);
  EXPECT_EQ(host_log[1].checksum, kExpectedChecksum64k);
  EXPECT_EQ(host_log[2].entry_id, kEntryId3);
  EXPECT_EQ(host_log[2].checksum, kExpectedChecksum128k);
}

TEST_F(ChecksumKernelTest, OverflowingChecksumsAreDiscarded) {
  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceMemory<se::cuda::SdcChecksumLogHeader> device_log,
      se::cuda::CreateSdcLogOnDevice(*stream_, /*max_entries=*/2));
  auto cleanup_log =
      absl::MakeCleanup([&]() { executor_->Deallocate(&device_log); });

  EXPECT_OK(AppendChecksumOnDevice(kEntryId1, kInput32k, device_log, kTestDim));
  EXPECT_OK(AppendChecksumOnDevice(kEntryId2, kInput64k, device_log, kTestDim));
  // This entry will be discarded.
  EXPECT_OK(
      AppendChecksumOnDevice(kEntryId3, kInput128k, device_log, kTestDim));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log,
                          se::cuda::ReadSdcLogFromDevice(*stream_, device_log));
  ASSERT_GE(host_log.size(), 2);
  EXPECT_EQ(host_log[0].entry_id, kEntryId1);
  EXPECT_EQ(host_log[0].checksum, kExpectedChecksum32k);
  EXPECT_EQ(host_log[1].entry_id, kEntryId2);
  EXPECT_EQ(host_log[1].checksum, kExpectedChecksum64k);
}

}  // namespace
