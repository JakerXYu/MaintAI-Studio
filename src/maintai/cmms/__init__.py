"""Mock CMMS subsystem (P1 vertical slice, no live connector).

Exposes the :class:`MockCMMSService` plus its stable error surface. The service
turns an already-approved ``cmms_work_order`` approval into exactly one mock
work order (status ``approved``) and one execution receipt, idempotently, without
contacting any live CMMS.
"""

from maintai.cmms.service import (
    MOCK_CMMS_DISCLAIMER,
    CMMSApprovalNotExecutableError,
    CMMSApprovalNotFoundError,
    CMMSPredictionEventNotFoundError,
    CMMSValidationError,
    MockCMMSService,
    MockCMMSServiceError,
)

__all__ = [
    "CMMSApprovalNotFoundError",
    "CMMSApprovalNotExecutableError",
    "CMMSPredictionEventNotFoundError",
    "CMMSValidationError",
    "MOCK_CMMS_DISCLAIMER",
    "MockCMMSService",
    "MockCMMSServiceError",
]
