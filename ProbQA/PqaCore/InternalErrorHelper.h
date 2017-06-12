#pragma once

#include "../PqaCore/Interface/PqaErrorParams.h"

#define MAKE_INTERR PqaError(PqaErrorCode::Internal, new InternalErrorParams(__FILE__, __LINE__))
#define MAKE_INTERR_MSG(msgVar) PqaError(PqaErrorCode::Internal, new InternalErrorParams(__FILE__, __LINE__), msgVar)
