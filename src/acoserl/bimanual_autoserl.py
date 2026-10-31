"""Public supervisor entry point; one retained baseline plus reviewed execution.

bimanual_core.py is the byte-for-byte pre-review baseline, not a second active
policy. All collectors import this class, so unused experimental Guide mixins
cannot silently replace the tested production path.
"""
from acoserl.bimanual_core import (
    ROOT, DEFAULT, SURVEY, ARMS, SafetyStop, ActionOnly, snapshot, poses,
    objects, Contacts, servo, errors, strict_geometry, direction_ok,
    LegacyBimanualAutoSERL, BimanualAutoSERL as _RetainedBaseline,
)
from acoserl.reviewed_runtime import ReviewedRuntimeMixin


class BimanualAutoSERL(ReviewedRuntimeMixin, _RetainedBaseline):
    """Single 14D Actor; independent Guide, qualified Recovery, real replay."""
    pass
