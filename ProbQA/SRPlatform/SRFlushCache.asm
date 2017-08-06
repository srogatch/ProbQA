; http://lallouslab.net/2016/01/11/introduction-to-writing-x64-assembly-in-visual-studio/
_DATA SEGMENT

_DATA ENDS

_TEXT SEGMENT

PUBLIC SRFlushCache

; https://docs.microsoft.com/en-us/cpp/build/overview-of-x64-calling-conventions
; RCX=pFirstCl
; RDX=pLimCl
; R8=clSize
SRFlushCache PROC

SRFlushCache_Loop:
  clflushopt byte ptr [RCX]
  add RCX, R8
  cmp RCX, RDX ; RCX-RDX
  jl SRFlushCache_Loop
  ret

SRFlushCache ENDP

_TEXT ENDS

END
