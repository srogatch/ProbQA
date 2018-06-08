using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace ProbQANetCore
{
  public class PqaError
  {
    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void CiReleasePqaError(IntPtr pError);

    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr CiPqaError_ToString(IntPtr pError, byte withParams);

    private IntPtr _nativeErr;

    private PqaError(IntPtr nativeErr)
    {
      _nativeErr = nativeErr;
    }

    ~PqaError()
    {
        CiReleasePqaError(_nativeErr);
    }

    public static PqaError Factor(IntPtr nativeErr)
    {
      if(nativeErr == IntPtr.Zero)
      {
        return null;
      }
      return new PqaError(nativeErr);
    }

    public override string ToString()
    {
      return ToString(true);
    }

    public string ToString(bool withParams)
    {
      if (_nativeErr == IntPtr.Zero)
      {
        return "Success";
      }
      return Utils.HandleNativeString(CiPqaError_ToString(_nativeErr, (byte)(withParams ? 1 : 0)));
    }
  }
}
