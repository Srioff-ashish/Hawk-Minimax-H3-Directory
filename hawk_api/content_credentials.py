"""Reading an image's C2PA content credential, to record where an AI image really came from.

Google signs every Nano Banana image with a C2PA manifest saying the picture was made by a generative model
(digitalSourceType "trainedAlgorithmicMedia"). An image like that, made outside this server and uploaded, arrives
with no history here, so the upload rules -- there because an uploaded photo may show a real person -- apply to
a picture that is AI-generated. When its credential checks out, HawkService.verify_ai_credential records the
true origin instead: generated, by the signer named in the credential, with the credential kept as evidence.

What a credential proves, and what it does not: a valid signature from a trusted issuer says that issuer's model
produced these pixels and that nothing has changed them since. It does not say what went into the model -- a
photo passed to Gemini as a reference comes back credentialed as AI output like any other, and Google's
manifest records only that an input existed. So a credential that lists inputs passes only when the caller
names exactly that many, each of them an AI-generated image in the library (HawkService checks that), and the
record names the issuer rather than claiming a prompt this server never saw.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field

#: Signers whose AI-generation claim is accepted. A credential from anyone else is reported, not trusted.
TRUSTED_ISSUERS = ("Google LLC",)
AI_SOURCE = "trainedAlgorithmicMedia"
#: Validation codes that do not undermine the claim: the signature is checked, but this server carries no C2PA
#: trust list, so the signing certificate is always reported as untrusted. The issuer check stands in for it.
TOLERATED_STATUS = frozenset({"signingCredential.untrusted"})


@dataclass(frozen=True)
class CredentialCheck:
    ok: bool
    reason: str
    issuer: str = ""
    generator: str = ""
    details: dict = field(default_factory=dict)


def read_manifest_store(data: bytes, mime: str) -> dict | None:
    """The C2PA manifest store embedded in this file, or None when it has none.

    Raises RuntimeError when the c2pa library is missing, so the caller can say so instead of reporting a
    file with a perfectly good credential as having none.
    """
    try:
        import c2pa  # c2pa-python; optional, so the rest of the API runs without it
    except ImportError:
        raise RuntimeError("Checking content credentials needs the c2pa-python package: pip install c2pa-python") from None
    try:
        with c2pa.Reader(mime or "image/jpeg", io.BytesIO(data)) as reader:
            return json.loads(reader.json())
    except Exception:  # c2pa raises for a file with no manifest, and for one it cannot parse
        return None


def evaluate(store: dict | None, inputs: int = 0) -> CredentialCheck:
    """Whether a manifest store shows a picture made entirely by a trusted issuer's generative model.

    ``inputs`` is how many input images the caller says went in. Google's credential records that an input was
    used but nothing about it, so the claim stands only when it names exactly that many, and the caller must
    check each of those is itself AI-generated -- otherwise this would pass an AI edit of a real photo.
    """
    if not store or not store.get("active_manifest"):
        return CredentialCheck(False, "The file carries no content credential (C2PA manifest).")
    manifest = (store.get("manifests") or {}).get(store["active_manifest"]) or {}
    issuer = str((manifest.get("signature_info") or {}).get("issuer") or "")
    generator = ", ".join(str(g.get("name") or "") for g in manifest.get("claim_generator_info") or []
                          if isinstance(g, dict) and g.get("name")) or str(manifest.get("claim_generator") or "")
    problems = [str(s.get("code") or "") for s in store.get("validation_status") or []
                if str(s.get("code") or "") not in TOLERATED_STATUS]
    if str(store.get("validation_state") or "") not in ("Valid", "Trusted") or problems:
        return CredentialCheck(False, "The content credential does not validate"
                               + (f" ({', '.join(problems)})." if problems else "."), issuer, generator)
    if issuer not in TRUSTED_ISSUERS:
        return CredentialCheck(False, f"The credential is signed by {issuer or 'an unnamed issuer'}, which is not one "
                               f"this server accepts ({', '.join(TRUSTED_ISSUERS)}).", issuer, generator)
    actions = [action for assertion in manifest.get("assertions") or []
               if str(assertion.get("label") or "").startswith("c2pa.actions")
               for action in (assertion.get("data") or {}).get("actions") or []]
    sources = [str(action.get("digitalSourceType") or "").rsplit("/", 1)[-1] for action in actions]
    if not any(action.get("action") == "c2pa.created" for action in actions) or not sources:
        return CredentialCheck(False, "The credential does not record how the picture was created.", issuer, generator)
    if any(source != AI_SOURCE for source in sources):
        # e.g. digitalCapture (a camera) or compositeWithTrainedAlgorithmicMedia (AI over a real photo)
        return CredentialCheck(False, "The credential says part of this picture did not come from a generative model "
                               f"({', '.join(sorted(set(sources) - {AI_SOURCE}) or ['unspecified'])}).", issuer, generator)
    ingredients = len(manifest.get("ingredients") or [])
    if ingredients != inputs:
        if not inputs:
            return CredentialCheck(False, f"The credential lists {ingredients} input image(s), so it may have been made "
                                   "from a photo. Name the image(s) it was made from (made_from), each one an "
                                   "AI-generated image in this library.", issuer, generator)
        return CredentialCheck(False, f"The credential lists {ingredients} input image(s), but {inputs} were named.",
                               issuer, generator)
    made = "from no input files" if not ingredients else f"from {ingredients} input image(s)"
    return CredentialCheck(True, f"Signed by {issuer}: made entirely by a generative model, {made}.",
                           issuer, generator,
                           {"issuer": issuer, "generator": generator, "digital_source_type": AI_SOURCE,
                            "actions": [str(action.get("action") or "") for action in actions],
                            "inputs": ingredients, "validation_state": store.get("validation_state")})


def check(data: bytes, mime: str, inputs: int = 0) -> CredentialCheck:
    return evaluate(read_manifest_store(data, mime), inputs)
