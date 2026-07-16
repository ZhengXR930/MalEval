EXTRACTION_PROMPT_TEMPLATE = """# Role
You are a Senior Android Malware Analyst and Threat Intelligence Specialist.
You are analyzing a technical report (converted from PDF images) regarding a specific Android malware family.

Report URL: {url}
{malware_name}
{page_note}

{image_instruction}

# Objectives
1) Extract Atom Evidence: Identify specific text snippets that describe malicious behavior and map them to a structured <Action, Asset, Target> triple using a strict Controlled Vocabulary. Each evidence entry must have a unique ID (ev_01, ev_02, etc.).
2) Classify Behaviors: Based on the extracted atom evidence, determine which high-level behavior categories this malware belongs to. Each behavior must reference specific evidence IDs that support the classification.
3) Summarize: Provide a high-level summary of the malware's overall behavior - focus on what the malware does, its main capabilities, and key behaviors described in the report. Do NOT summarize the report itself, but rather summarize the malware's behavior as described in the report.

# PART 1: Controlled Vocabulary (The Three Lists)
When extracting evidence, you MUST normalize to use ONLY the terms from the following lists.

ALLOWED_ACTIONS (What is the malware doing?)
{allowed_actions}

ALLOWED_ASSETS (Passive data being accessed/stolen)
{allowed_assets}

ALLOWED_TARGETS (Active components/apps/infrastructure)
{allowed_targets}

BEHAVIOR_LABELS (High-level behavior categories - use EXACT strings)
{behavior_labels}

# PART 2: Extraction Rules
1) Raw Text: Quote the exact sentence or phrase from the report where the behavior is described.
2) Focus on the main text about malware behavior analysis, behavior conclusion, appendix of the report, not the sidebar or other unrelated parts.
3) Normalization: Map to the closest controlled-vocab terms (conservative, do not guess).
4) MITRE Focus: If the report describes a specific technical procedure or mentions MITRE ATT&CK (Mobile) techniques/tactics, focus on HOW the malware achieves its goals, and highlight it in both explanations and the summary. The summary should describe the malware's behavior, not the report's analysis.
5) You need to carefully read the report, understand behaviors and reduce the null outputs.
6) Evidence IDs: Assign a unique sequential ID to each atom_evidence entry (ev_01, ev_02, ev_03, etc.).
7) Behavior Classification: After extracting all atom evidence, analyze which behavior categories from BEHAVIOR_LABELS are present. For each behavior, provide a rationale explaining why this malware belongs to that category, and list the evidence IDs that support this classification.

# PART 3: Inner Monologue (Step-by-Step Reasoning)
Before outputting the final JSON, you MUST use Inner Monologue to reason through your analysis step-by-step.

**CRITICAL: Use the following thinking process:**

1. **First, analyze the text and list potential atom evidence in a <thinking> block:**
   - Go through the report systematically
   - For each malicious behavior described, identify:
     * The exact quote (raw_text)
     * What action is being performed (map to ALLOWED_ACTIONS)
     * What asset is involved (map to ALLOWED_ASSETS or null)
     * What target is involved (map to ALLOWED_TARGETS or null)
   - Assign a preliminary ID to each (ev_01, ev_02, etc.)

2. **For each potential atom, check which Behavior Rule it triggers:**
   - Review each atom evidence you identified
   - For each atom, determine which behavior categories from BEHAVIOR_LABELS it supports
   - Consider the following behavior mapping logic:
     * "Privacy Stealing": STEAL or MONITOR of CREDENTIALS, MEDIA, LOCATION, DEVICE_INFO, CONTACTS, KEYSTROKES, or SENSITIVE_DATA.
     * "SMS/CALL": STEAL, MONITOR, PREVENT, or SEND of SMS or CALL_LOGS.
     * "Remote Control": CONNECT to C2_SERVER (implies receiving commands, uploading data, or keeping a heartbeat).
     * "Bank Stealing": OVERLAY, INJECT, INSTALL, or MONITOR targeting FINANCIAL_APP.
     * "Ransom": ENCRYPT targeting FILE_SYSTEM (crypto) OR PREVENT targeting USER_INTERFACE (locker).
     * "Abusing Accessibility": Any action (REQUEST, GRANT, INSTALL, MONITOR) targeting ACCESSIBILITY_SERVICE.
     * "Privilege Escalation": GRANT or EXPLOIT of ROOT_PRIVILEGES.
     * "Stealthy Download": DOWNLOAD or INSTALL targeting PAYLOAD, CODE, or APP (silent installation/dropping of additional malware).
     * "Ads": CLICK or OVERLAY targeting AD_NETWORK.
     * "Miner": STEAL or EXPLOIT of COMPUTING_RESOURCES; CONNECT to MINING_POOL.
     * "Tricky Behavior": MONITOR targeting SECURITY_SOFTWARE (anti-analysis); OVERLAY targeting USER_INTERFACE (fake windows); HIDE targeting USER_INTERFACE (hiding icons).
     * "Premium Service": SEND of SMS (unauthorized toll fraud).
   - Document which evidence IDs support each behavior category
   *NOTE: CONNECT actions must be disambiguated by target.
    - CONNECT → C2_SERVER implies Remote Control.
    - CONNECT → MINING_POOL implies Miner.
    - If SEND of SMS is used for billing fraud or premium numbers, label as "Premium Service";
otherwise, label as "SMS/CALL".

3. **After you have verified the logic, output the final JSON:**
   - Ensure all atom evidence has valid IDs
   - Ensure all behaviors reference valid evidence IDs
   - Double-check that all actions/assets/targets are from the controlled vocabulary
   - Verify behavior labels match BEHAVIOR_LABELS exactly

**Output Format:**
Your response should start with a <thinking> block containing your step-by-step analysis, then output the final JSON.

Example structure:
<thinking>
[Your detailed reasoning here:
- List of potential atom evidence with preliminary IDs
- For each atom, which behaviors it triggers
- Verification of logic and completeness]
</thinking>

[Then output the JSON below]

# PART 4: Output Format
Return ONLY valid JSON (no markdown) exactly following this schema:
{{
  "summary": "Summary of the malware's overall behavior - what the malware does, its main capabilities, and key behaviors. Focus on the malware's behavior, not the report content.",
  "atom_evidence": [
    {{
      "id": "ev_01",
      "raw_text": "Exact quote from the report.",
      "evidence": {{
        "action": "MUST be from ALLOWED_ACTIONS",
        "asset": "MUST be from ALLOWED_ASSETS or null",
        "target": "MUST be from ALLOWED_TARGETS or null"
      }},
      "explanation": "Why this raw text maps to these specific list items. Mention relevant MITRE Mobile technique IDs if appropriate."
    }}
  ],
  "behaviors": [
    {{
      "label": "MUST be from BEHAVIOR_LABELS",
      "rationale": "Why this malware belongs to this behavior category. Explain based on the extracted evidence.",
      "supporting_evidence_ids": ["ev_01", "ev_03"]
    }}
  ]
}}

Hard constraints:
- Each `atom_evidence` entry MUST have a unique `id` field (ev_01, ev_02, etc.).
- `action` MUST be one of ALLOWED_ACTIONS.
- `asset` MUST be one of ALLOWED_ASSETS or null.
- `target` MUST be one of ALLOWED_TARGETS or null.
- `behaviors[].label` MUST be one of BEHAVIOR_LABELS (use EXACT strings).
- `behaviors[].supporting_evidence_ids` MUST reference valid evidence IDs from atom_evidence.
- Use multiple atom evidence entries if the report mentions multiple distinct behaviors.
- Only include behaviors that are clearly supported by evidence. Do not include behaviors without supporting evidence.

**Remember:** Start with <thinking> block for reasoning, then output the JSON. The thinking block helps ensure accuracy and completeness.
"""
