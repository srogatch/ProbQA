// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace ProbQA {

template<typename taNumber> class TargetRowPersistence {
  SRPlat::SRSmartFile *_pSf;
  const TPqaId _nTargets;
public:
  TargetRowPersistence() : _pSf(nullptr), _nTargets(-1) { }
  TargetRowPersistence(SRPlat::SRSmartFile &sf, const TPqaId nTargets) : _pSf(&sf), _nTargets(nTargets) { }
  template<bool taCD> bool Write(const SRPlat::SRFastArray<taNumber, taCD>& source);
  template<bool taCD> bool Read(SRPlat::SRFastArray<taNumber, taCD>& dest);
};

template<typename taNumber> template<bool taCD> bool TargetRowPersistence<taNumber>::Write(
  const SRPlat::SRFastArray<taNumber, taCD>& source)
{
  return TPqaId(std::fwrite(source.Get(), sizeof(taNumber), _nTargets, _pSf->Get())) == _nTargets;
}

template<typename taNumber> template<bool taCD> bool TargetRowPersistence<taNumber>::Read(
  SRPlat::SRFastArray<taNumber, taCD>& dest)
{
  return TPqaId(std::fread(dest.Get(), sizeof(taNumber), _nTargets, _pSf->Get())) == _nTargets;
}

} // namespace ProbQA
