// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "../SRPlatform/Interface/SRMemPool.h"

namespace SRPlat {

SRBaseMemPool gBaseMemPool;

SRBaseMemPool& SRGetBaseMemPool() {
  return gBaseMemPool;
}
  
} // namespace SRPlat
