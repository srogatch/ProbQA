// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CpuEngine.decl.h"

namespace ProbQA {

template<typename taNumber> inline const taNumber&
CpuEngine<taNumber>::GetA(const TPqaId iQuestion, const TPqaId iAnswer, const TPqaId iTarget) const {
  return _sA[SRPlat::SRCast::ToSizeT(iQuestion)][SRPlat::SRCast::ToSizeT(iAnswer)][SRPlat::SRCast::ToSizeT(iTarget)];
}
template<typename taNumber> inline taNumber&
CpuEngine<taNumber>::ModA(const TPqaId iQuestion, const TPqaId iAnswer, const TPqaId iTarget) {
  return _sA[SRPlat::SRCast::ToSizeT(iQuestion)][SRPlat::SRCast::ToSizeT(iAnswer)][SRPlat::SRCast::ToSizeT(iTarget)];
}

template<typename taNumber> inline const taNumber&
CpuEngine<taNumber>::GetD(const TPqaId iQuestion, const TPqaId iTarget) const {
  return _mD[SRPlat::SRCast::ToSizeT(iQuestion)][SRPlat::SRCast::ToSizeT(iTarget)];
}
template<typename taNumber> inline taNumber&
CpuEngine<taNumber>::ModD(const TPqaId iQuestion, const TPqaId iTarget) {
  return _mD[SRPlat::SRCast::ToSizeT(iQuestion)][SRPlat::SRCast::ToSizeT(iTarget)];
}

template<typename taNumber> inline const taNumber&
CpuEngine<taNumber>::GetB(const TPqaId iTarget) const {
  return _vB[SRPlat::SRCast::ToSizeT(iTarget)];
}
template<typename taNumber> inline taNumber&
CpuEngine<taNumber>::ModB(const TPqaId iTarget) {
  return _vB[SRPlat::SRCast::ToSizeT(iTarget)];
}

} // namespace ProbQA
