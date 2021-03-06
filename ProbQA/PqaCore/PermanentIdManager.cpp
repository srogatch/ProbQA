// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/PermanentIdManager.h"

using namespace SRPlat;

namespace ProbQA {

TPqaId PermanentIdManager::PermFromComp(const TPqaId compId) {
  if (compId < 0 || compId >= TPqaId(_comp2perm.size())) {
    return cInvalidPqaId;
  }
  return _comp2perm[compId];
}

TPqaId PermanentIdManager::CompFromPerm(const TPqaId permId) {
  auto it = _perm2comp.find(permId);
  if (it == _perm2comp.end()) {
    return cInvalidPqaId;
  }
  return it->second;
}

bool PermanentIdManager::Save(FILE *fpout, const bool empty) {
  const TPqaId nComp = (empty ? 0 : _comp2perm.size());
  if (fwrite(&_nextPermId, sizeof(_nextPermId), 1, fpout) != 1) {
    return false;
  }
  if (fwrite(&nComp, sizeof(nComp), 1, fpout) != 1) {
    return false;
  }
  if (TPqaId(fwrite(_comp2perm.data(), sizeof(_comp2perm[0]), nComp, fpout)) != nComp) {
    return false;
  }
  return true;
}

bool PermanentIdManager::Load(FILE *fpin) {
  TPqaId nComp;
  if (fread(&_nextPermId, sizeof(_nextPermId), 1, fpin) != 1) {
    return false;
  }
  if (fread(&nComp, sizeof(nComp), 1, fpin) != 1) {
    return false;
  }
  _comp2perm.resize(nComp);
  _perm2comp.clear();
  if (TPqaId(fread(_comp2perm.data(), sizeof(_comp2perm[0]), nComp, fpin)) != nComp) {
    return false;
  }
  for (TPqaId i = 0; i < nComp; i++) {
    if (_comp2perm[i] == cInvalidPqaId) {
      continue;
    }
    _perm2comp.emplace(_comp2perm[i], i);
  }
  return true;
}

bool PermanentIdManager::EnsurePermIdGreater(const TPqaId bound) {
  if (_nextPermId <= bound) {
    _nextPermId = bound + 1;
    return true;
  }
  return false;
}

bool PermanentIdManager::RemoveComp(const TPqaId compId) {
  if (compId >= TPqaId(_comp2perm.size())) {
    SRUtils::RequestDebug();
    return false;
  }
  const TPqaId iPerm = _comp2perm[compId];
  if (iPerm == cInvalidPqaId) {
    SRUtils::RequestDebug();
    return false;
  }
  auto it = _perm2comp.find(iPerm);
  if (it == _perm2comp.end()) {
    //TODO: actually, this is inconsistency in the manager itself - consider throwing an exception
    SRUtils::RequestDebug();
    return false;
  }
  _perm2comp.erase(it);
  _comp2perm[compId] = cInvalidPqaId;
  return true;
}

bool PermanentIdManager::RenewComp(const TPqaId compId) {
  if (compId >= TPqaId(_comp2perm.size())) {
    SRUtils::RequestDebug();
    return false; // out of range for compact IDs on record
  }
  if (_comp2perm[compId] != cInvalidPqaId) {
    // Compact ID is already mapped to a permanent ID. Use RemoveComp() first.
    SRUtils::RequestDebug();
    return false;
  }
  _comp2perm[compId] = _nextPermId;
  _perm2comp.emplace(_nextPermId, compId);
  _nextPermId++;
  return true;
}

bool PermanentIdManager::GrowTo(const TPqaId nComp) {
  if (nComp < TPqaId(_comp2perm.size())) {
    SRUtils::RequestDebug();
    return false;
  }
  for (TPqaId i = _comp2perm.size(); i < nComp; i++) {
    _comp2perm.push_back(_nextPermId);
    _perm2comp.emplace(_nextPermId, i);
    _nextPermId++;
  }
  return true;
}

bool PermanentIdManager::OnCompact(const TPqaId nNew, const TPqaId *pOldIds) {
  if (nNew > TPqaId(_comp2perm.size())) {
    SRUtils::RequestDebug();
    return false;
  }
  if (nNew != TPqaId(_perm2comp.size())) {
    SRUtils::RequestDebug();
    return false;
  }
  for (TPqaId i = 0; i < nNew; i++) {
    const TPqaId oldComp = pOldIds[i];
    if (oldComp < 0 || oldComp >= TPqaId(_comp2perm.size())) {
      SRUtils::RequestDebug();
      return false;
    }
    const TPqaId oldPerm = _comp2perm[oldComp];
    if (oldPerm == cInvalidPqaId) {
      //TODO: actually, this is inconsistency in the manager itself - consider throwing an exception
      SRUtils::RequestDebug();
      return false;
    }
    auto it = _perm2comp.find(oldPerm);
    if (it == _perm2comp.end()) {
      //TODO: actually, this is inconsistency in the manager itself - consider throwing an exception
      SRUtils::RequestDebug();
      return false;
    }
    it->second = i;
  }

  _comp2perm.clear();
  _comp2perm.resize(nNew, cInvalidPqaId);
  for (std::pair<TPqaId, TPqaId> m : _perm2comp) {
    if (m.second < 0 || m.second >= nNew) {
      //TODO: actually, this is inconsistency in the manager itself - consider throwing an exception
      SRUtils::RequestDebug();
      return false;
    }
    _comp2perm[m.second] = m.first;
  }

  return true;
}

bool PermanentIdManager::RemapPermId(const TPqaId srcPermId, const TPqaId destPermId) {
  if (destPermId >= _nextPermId) {
    return false; // can't remap to future permId's
  }
  auto jt = _perm2comp.find(destPermId);
  if (jt != _perm2comp.end()) {
    return false; // there's already such permId on record
  }
  auto it = _perm2comp.find(srcPermId);
  if (it == _perm2comp.end()) {
    return false; // no such permId on record
  }
  const TPqaId compId = it->second;
  _perm2comp.erase(it);
  _perm2comp.emplace(destPermId, compId);
  _comp2perm[compId] = destPermId;
  return true;
}

} // namespace ProbQA
