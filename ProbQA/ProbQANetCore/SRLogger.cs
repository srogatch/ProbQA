using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace ProbQANetCore
{
  public class SRLogger
  {
    [DllImport("PqaCore.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern byte CiLogger_Init(out IntPtr pStrErr, string baseName);

    public static bool Init(string baseName)
    {
      IntPtr pStrErr = IntPtr.Zero;
      bool retVal = (CiLogger_Init(out pStrErr, baseName) != 0);
      if(pStrErr != IntPtr.Zero)
      {
        throw new Exception("C++ engine returned [" + retVal + "]: " + Utils.HandleNativeString(pStrErr));
      }
      return retVal;
    }
  }
}
