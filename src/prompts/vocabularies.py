MALWARE_TYPES = [
    "Spyware", "Ransomware", "Adware", "Banker", "Trojan", 
    "Downloader", "Miner", "Scareware", "Rootkit", "Botnet"
]

MALWARE_TYPES_SIMPLE = [
    "adware", "spyware", "ransomware", "banker", "rootkit", "trojan"
]

MALWARE_TYPE_SELECTION_RULES_SIMPLE = """
Type selection rules:
- Pick `type_name` by the dominant malicious objective; prefer the most specific supported type.
- `banker`: financial fraud, banking/payment credential theft, account takeover, transaction manipulation, or SMS/OTP interception for unauthorized financial operations.
- `spyware`: covert surveillance or persistent collection/exfiltration of private user, device, communication, or location data, when financial fraud is not the dominant objective.
- `ransomware`: extortion through file encryption, device/screen locking, access restriction, or threats of data loss/disclosure.
- `adware`: abusive ad monetization, including intrusive/deceptive ads, forced redirects, unauthorized ad clicks, or ad-traffic manipulation.
- `rootkit`: privileged access used for stealth, persistence, system-level compromise, or hiding malicious files, processes, services, or components.
- `trojan`: fallback only when malware is clear but evidence supports only generic deception, payload delivery, unauthorized access, or broad compromise without a more specific dominant type.
""".strip()

BEHAVIOR_LABELING_GUARDRAIL = (
    "Assign a behavior label only when it is directly supported by the required "
    "action/asset/target evidence pattern; do not infer higher-level behaviors "
    "from generic obfuscation, reflection, dynamic loading, or code injection alone."
)

BEHAVIOR_LABELS = [
    "Privacy Stealing", "SMS/CALL", "Remote Control", "Bank Stealing", 
    "Ransom", "Abusing Accessibility", "Privilege Escalation", 
    "Stealthy Download", "Ads", "Miner", "Tricky Behavior", "Premium Service"
]

ACTIONS = [
    "STEAL", "DOWNLOAD", "INSTALL", "HIDE", "OVERLAY", "CLICK", 
    "ENCRYPT", "CONNECT", "PREVENT", "REQUEST", "GRANT", "SEND", 
    "INJECT", "MONITOR", "CAPTURE", "EXPLOIT"
]

ASSETS = [
    "CREDENTIALS", "FINANCIAL_DATA", "SMS", "CALL_LOGS", "MEDIA", 
    "LOCATION", "DEVICE_INFO", "NOTIFICATIONS", "CLIPBOARD", 
    "CONTACTS", "KEYSTROKES", "SENSITIVE_DATA", "APP", "PAYLOAD", 
    "ROOT_PRIVILEGES", "ADMIN_PRIVILEGES", "CODE", "UI_ELEMENT", 
    "COMPUTING_RESOURCES", "null"
]

TARGETS = [
    "FINANCIAL_APP", "SOCIAL_APP", "SYSTEM_SETTINGS", "C2_SERVER", 
    "AD_NETWORK", "USER_INTERFACE", "SECURITY_SOFTWARE", 
    "ACCESSIBILITY_SERVICE", "BROWSER", "DEVICE_ADMIN", 
    "HARDWARE_SENSOR", "FILE_SYSTEM", "MINING_POOL", "null"
]
