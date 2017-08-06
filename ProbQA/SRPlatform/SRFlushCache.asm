_DATA SEGMENT

_DATA ENDS

_TEXT SEGMENT

PUBLIC SRFlushCache

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
