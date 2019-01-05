// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace ProbQA {

// This class is not thread-safe. The underlying bit array must set to "true" (indicating a gap) values beyond the size,
//   so to allow uniform handling of partial vectors on the bounds as if the invalid items are just in a gap.
template <typename taId> class GapTracker {
  std::vector<taId> _gaps;
  SRPlat::SRBitArray _isGap; // remove R after refactorring
public:
  explicit GapTracker() : _isGap(0, true) { }

  bool IsGap(const taId at) const {
    assert(0 <= at && at < taId(_isGap.Size()));
    return _isGap.GetOne(SRPlat::SRCast::ToUint64(at));
  }

  // Get |iQuad|th 4 adjacent bits denoting gaps.
  uint8_t GetQuad(const taId iQuad) const { return _isGap.GetQuad(iQuad); }

  template<typename taResult> const taResult& GetPacked(const taId iPack) const {
    return _isGap.GetPacked<taResult>(iPack);
  }

  void Release(const taId at) {
    assert(!_isGap.GetOne(SRPlat::SRCast::ToUint64(at)));
    _isGap.SetOne(SRPlat::SRCast::ToUint64(at));
    _gaps.push_back(at);
  }

  bool HasGaps() const { return _gaps.size() >= 1; }

  // If there is no gap to acquire, returns length+1 meaning the client must increase his arrays.
  taId Acquire() {
    if (_gaps.size() <= 0) {
      taId answer = _isGap.Size();
      _isGap.Add(1);
      _isGap.ClearOne(answer);
      return answer;
    }
    taId answer = _gaps.back();
    _gaps.pop_back();
    _isGap.ClearOne(SRPlat::SRCast::ToUint64(answer));
    return answer;
  }

  // Grow to the specified length setting the new items as not gaps.
  void GrowTo(const taId newLength) {
    assert(newLength >= taId(_isGap.Size()));
    _isGap.GrowTo(SRPlat::SRCast::ToUint64(newLength), false);
  }

  // Client access to the gaps vector, e.g. to sort it and then make compaction.
  std::vector<taId>& Gaps() { return _gaps; }

  taId GetNGaps() const { return _gaps.size(); }
  const taId* ListGaps() const { return _gaps.data(); }
  const void* GetBits() const { return _isGap.Data(); }

  void Compact(const taId newLength) {
    assert(newLength <= taId(_isGap.Size()));
    _isGap.ClearRange(0, SRPlat::SRCast::ToUint64(newLength));
    _isGap.ReduceTo(newLength);
    _gaps.clear();
  }
};

} // namespace ProbQA
