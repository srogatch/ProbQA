#include "stdafx.h"
#include "../PqaCore/CudaPersistence.h"
#include "../PqaCore/CudaMacros.h"
#include "../PqaCore/PqaException.h"

using namespace SRPlat;

namespace ProbQA {

CudaPersistence::CudaPersistence(const size_t bufSize, KBFileInfo *pKbFi, cudaStream_t cuStr)
  : _bufSize(bufSize), _pKbFi(pKbFi), _cuStr(cuStr), _buffer(new uint8_t[bufSize])
{ }


void CudaPersistence::ReadFile(void *d_p, size_t nBytes) {
  while (nBytes > 0) {
    const size_t nToRead = std::min(nBytes, _bufSize);
    const size_t nRead = fread(_buffer.get(), 1, nToRead, _pKbFi->_sf.Get());
    if (nToRead != nRead) {
      PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(_pKbFi->_filePath), SRString::MakeUnowned(
        SR_FILE_LINE " Can't read file.")).ThrowMoving();
    }
    //TODO: cudaHostRegister(cudaHostAllocPortable) to let copying be really asynchronous
    CUDA_MUST(cudaMemcpyAsync(d_p, _buffer.get(), nRead, cudaMemcpyHostToDevice, _cuStr));
    d_p = static_cast<uint8_t*>(d_p) + nRead;
    nBytes -= nRead;
  }
}

void CudaPersistence::WriteFile(const void *d_p, size_t nBytes) {
  while (nBytes > 0) {
    const size_t nToWrite = std::min(nBytes, _bufSize);
    //TODO: cudaHostRegister(cudaHostAllocPortable) to let copying be really asynchronous
    CUDA_MUST(cudaMemcpyAsync(_buffer.get(), d_p, nToWrite, cudaMemcpyDeviceToHost, _cuStr));
    const size_t nWritten = fwrite(_buffer.get(), 1, nToWrite, _pKbFi->_sf.Get());
    if (nWritten != nToWrite) {
      PqaException(PqaErrorCode::FileOp, new FileOpErrorParams(_pKbFi->_filePath), SRString::MakeUnowned(
        SR_FILE_LINE " Can't write file.")).ThrowMoving();
    }
    d_p = static_cast<const uint8_t*>(d_p) + nWritten;
    nBytes -= nWritten;
  }
}

} // namespace ProbQA
