// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace SRPlat {

class SRStringUtils {
public:
  static std::string Escape(const std::string& src) {
    std::string ans;
    for (size_t i = 0; i < src.size(); i++) {
      if (src[i] == '\\') {
        ans.push_back(src[i]);
      }
      ans.push_back(src[i]);
    }
    return std::move(ans);
  }
};

} // namespace SRPlat
