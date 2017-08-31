// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace ProbQA {

// This class is not thread-safe.
//TODO: refactor to get a possibility to fetch 4 bits at once
template <typename taId> class GapTracker {
  std::vector<taId> _gaps;
  std::vector<bool> _isGap;
public:
  explicit GapTracker() { }

  bool IsGap(const taId at) const { return _isGap[SRCast::ToSizeT(at)]; }

  void Release(const taId at) {
    assert(!_isGap[at]);
    _isGap[SRCast::ToSizeT(at)] = true;
    _gaps.push_back(at);
  }

  bool HasGaps() const { return _gaps.size() >= 1; }

  // If there is no gap to acquire, returns length+1 meaning the client must increase his arrays.
  taId Acquire() {
    if (_gaps.size() <= 0) {
      taId answer = _isGap.size();
      _isGap.push_back(false);
      return answer;
    }
    taId answer = _gaps.back();
    _gaps.pop_back();
    _isGap[SRCast::ToSizeT(answer)] = false;
    return answer;
  }

  // Grow to the specified length setting the new items as not gaps.
  void GrowTo(const taId newLength) {
    assert(newLength >= taId(_isGap.size()));
    _isGap.resize(size_t(newLength), false);
  }

  // Client access to the gaps vector, e.g. to sort it and then make compaction.
  std::vector<taId>& Gaps() { return _gaps; }

  void Compact(const taId newLength) {
    _isGap.assign(SRCast::ToSizeT(newLength), false);
    _gaps.clear();
  }
};

} // namespace ProbQA
