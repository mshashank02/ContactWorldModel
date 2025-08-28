import hashlib
def stable_env_id(xml_path: str) -> str:
    return f"HandManipulateBlock_{hashlib.sha1(xml_path.encode()).hexdigest()[:8]}-v1"
