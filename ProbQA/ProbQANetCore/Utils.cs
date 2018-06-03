using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace ProbQANetCore
{
  internal class Utils
  {
    [DllImport("PqaCore.dll")]
    private static extern void CiReleaseString(IntPtr pString);

    // Take char*, convert to C# and release C++ memory.
    public static string HandleNativeString(IntPtr native)
    {
      string str = Marshal.PtrToStringAnsi(native);
      CiReleaseString(native);
      return str;
    }
  }
}
