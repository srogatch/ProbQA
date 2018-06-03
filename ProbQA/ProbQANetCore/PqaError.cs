using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace ProbQANetCore
{
  public class PqaError
  {
    [DllImport("PqaCore.dll")]
    private static extern void CiReleasePqaError(IntPtr pError);

    [DllImport("PqaCore.dll")]
    private static extern IntPtr CiPqaError_ToString(IntPtr pError, bool withParams);

    private IntPtr _native;

    internal PqaError(IntPtr native)
    {
      _native = native;
    }

    ~PqaError()
    {
      if(_native != IntPtr.Zero)
      {
        CiReleasePqaError(_native);
      }
    }

    bool IsOk { get { return _native == IntPtr.Zero; } }

    string ToString(bool withParams)
    {
      if (_native == IntPtr.Zero)
      {
        return "Success";
      }
      return Utils.HandleNativeString(CiPqaError_ToString(_native, withParams));
    }
  }
}
