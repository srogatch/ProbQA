#pragma once

#include "../PqaCore/Interface/PqaErrorParams.h"

#define MAKE_INTERR PqaError(PqaErrorCode::Internal, new InternalErrorParams(__FILE__, __LINE__))
#define MAKE_INTERR_MSG(msgVar) PqaError(PqaErrorCode::Internal, new InternalErrorParams(__FILE__, __LINE__), msgVar)

#define CATCH_TO_ERR_RETURN                                       \
  catch (SRException &ex) {                                       \
    return std::move(PqaError().SetFromException(std::move(ex))); \
  }                                                               \
  catch (std::exception &ex) {                                    \
    return std::move(PqaError().SetFromException(ex));            \
  }

#define CATCH_TO_ERR_SET(errVar)            \
  catch (SRException &ex) {                 \
    errVar.SetFromException(std::move(ex)); \
  }                                         \
  catch (std::exception& ex) {              \
    errVar.SetFromException(ex);            \
  }
