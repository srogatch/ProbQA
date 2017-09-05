// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace SRPlat {

template<typename... Ts> struct SRMaxSizeof {
  static constexpr size_t value = 0;
};

template<typename T, typename... Ts> struct SRMaxSizeof<T, Ts...> {
  static constexpr size_t value = std::max(sizeof(T), typename SRMaxSizeof<Ts...>::value);
};

} // namespace SRPlat