// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

class PermanentIdManager {
public: // methods
  // Get permanent ID from compact ID
  TPqaId PermFromComp(const TPqaId compId);
  TPqaId CompFromPerm(const TPqaId permId);

  bool Save(FILE *fpout, const bool empty=false);
  bool Load(FILE *fpin);
  bool EnsurePermIdGreater(const TPqaId bound);

  bool RemoveComp(const TPqaId compId);
  bool RenewComp(const TPqaId compId);

  // Grow, allocating permanent IDs
  bool GrowTo(const TPqaId nComp);
  bool OnCompact(const TPqaId nNew, const TPqaId *pOldIds);

private: // variables
  std::unordered_map<TPqaId, TPqaId> _perm2comp;
  std::vector<TPqaId> _comp2perm;
  TPqaId _nextPermId = 0;
};

} // namespace ProbQA
