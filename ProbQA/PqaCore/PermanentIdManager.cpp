// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/PermanentIdManager.h"

using namespace SRPlat;

namespace ProbQA {

TPqaId PermanentIdManager::PermFromComp(const TPqaId compId) {
  if (compId >= TPqaId(_comp2perm.size())) {
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

bool PermanentIdManager::Save(FILE *fpout) {
  const TPqaId nComp = _comp2perm.size();
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

bool PermanentIdManager::RemoveComp(const TPqaId compId) {
  if (compId >= TPqaId(_comp2perm.size())) {
    return false;
  }
  const TPqaId iPerm = _comp2perm[compId];
  if (iPerm == cInvalidPqaId) {
    return false;
  }
  auto it = _perm2comp.find(iPerm);
  if (it == _perm2comp.end()) {
    //TODO: actually, this is inconsistency in the manager itself - consider throwing an exception
    return false;
  }
  _perm2comp.erase(it);
  _comp2perm[compId] = cInvalidPqaId;
  return true;
}

bool PermanentIdManager::GrowTo(const TPqaId nComp) {
  if (nComp < TPqaId(_comp2perm.size())) {
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
    return false;
  }
  if (nNew != TPqaId(_perm2comp.size())) {
    return false;
  }
  for (TPqaId i = 0; i < nNew; i++) {
    const TPqaId oldComp = pOldIds[i];
    if (oldComp < 0 || oldComp >= TPqaId(_comp2perm.size())) {
      return false;
    }
    const TPqaId oldPerm = _comp2perm[oldComp];
    if (oldPerm == cInvalidPqaId) {
      //TODO: actually, this is inconsistency in the manager itself - consider throwing an exception
      return false;
    }
    auto it = _perm2comp.find(oldPerm);
    if (it == _perm2comp.end()) {
      //TODO: actually, this is inconsistency in the manager itself - consider throwing an exception
      return false;
    }
    it->second = i;
  }

  _comp2perm.clear();
  _comp2perm.resize(nNew, cInvalidPqaId);
  for (std::pair<TPqaId, TPqaId> m : _perm2comp) {
    if (m.second < 0 || m.second >= nNew) {
      //TODO: actually, this is inconsistency in the manager itself - consider throwing an exception
      return false;
    }
    _comp2perm[m.second] = m.first;
  }

  return true;
}

} // namespace ProbQA
