// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_DEPS_FILE_HELPERS_H_
#define MEDIAPIPE_DEPS_FILE_HELPERS_H_

#include "absl/strings/match.h"
#include "mediapipe/framework/deps/status.h"

namespace mediapipe {
namespace file {
::mediapipe::Status GetContents(absl::string_view file_name,
                                std::string* output);

::mediapipe::Status SetContents(absl::string_view file_name,
                                absl::string_view content);

::mediapipe::Status MatchInTopSubdirectories(
    const std::string& parent_directory, const std::string& file_name,
    std::vector<std::string>* results);

::mediapipe::Status Exists(absl::string_view file_name);

}  // namespace file
}  // namespace mediapipe

#endif  // MEDIAPIPE_DEPS_FILE_HELPERS_H_
