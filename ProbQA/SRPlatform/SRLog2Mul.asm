; http://lallouslab.net/2016/01/11/introduction-to-writing-x64-assembly-in-visual-studio/
_DATA SEGMENT

_DATA ENDS

_TEXT SEGMENT

PUBLIC SRLog2MulD

; http://www.website.masmforum.com/tutorials/fptute/fpuchap11.htm : description of FYL2X
; https://docs.microsoft.com/en-us/cpp/build/overview-of-x64-calling-conventions : "The x87 register stack is unused.
;   It may be used by the callee, but must be considered volatile across function calls."
; XMM0L=toLog
; XMM1L=toMul
SRLog2MulD PROC
  movq qword ptr [rsp+16], xmm1
  movq qword ptr [rsp+8], xmm0
  fld qword ptr [rsp+16]
  fld qword ptr [rsp+8]
  fyl2x
  fstp qword ptr [rsp+8]
  movq xmm0, qword ptr [rsp+8]
  ret

SRLog2MulD ENDP

_TEXT ENDS

END
