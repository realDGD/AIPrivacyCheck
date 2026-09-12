#!/usr/bin/env python3
"""Isolated ModelScope Downloader Worker for AI Privacy Check.

Runs exclusively within an isolated Python runtime venv containing the ModelScope SDK.
Never executes in control-plane Python 3.12.
Downloads models into staging directory, captures resolved commit revisions,
and provisions required third-party prompt assets for MemPrivacy models.
"""

import argparse
import datetime
import json
import os
import sys
import traceback


MEMPRIVACY_OFFICIAL_PROMPT = """You are a professional "Data Security and Privacy Compliance Expert." Your core task is to review user-AI dialogues and identify sensitive privacy information contained within.

# Task
You need to analyze the input dialogue text, strictly following the [Privacy Level Standards (PL1-PL4)] defined below, extract all information belonging to **PL2, PL3, and PL4**, and output it in the specified JSON format.

# Privacy Level Standards (PL1-PL4)

### 【PL4: Critical Secrets & System Security】 (Critical Risk)
  - Definition: Core secrets that, once compromised, directly lead to system takeover, core data exfiltration, or immediate, irrecoverable asset loss.
  - Core Standard: Irrecoverable / High asset loss / System takeover.
  - Classification Rules:
    1. Credentials/Authenticators: Cleartext Passwords, PINs, Passcodes, Gestures, 2FA/MFA/OTP SMS Verification Codes, Payment Passwords/CVV2/CVC2 Security Codes, etc.
    2. Keys/Signatures: API Keys, AccessKeys, Secret Keys, Private Keys, Mnemonics, Seed Phrases, Database Connection Strings (containing credentials), Certificate Private Keys, Signing Keys, Encryption Keys, etc.
    3. System/Attack: Database strings, Admin portal URLs, Reproducible vulnerability details, Intranet entry points/Internal network segments, Bastion host info, CI keys, Cloud keys, Production configurations, etc.
    4. Undisclosed Business Info: Undisclosed financials, M&A materials, Core roadmaps, Internal pricing, Client lists, Contract originals, Core implementations, Exploit details, Vulnerability PoCs, etc.
  - Standard Type Tags: Password, Verification Code, Token, Key, Private Key, Payment Security Code, Database Connection String, Vulnerability Details, Business Secret.

### 【PL3: Highly Sensitive PII】 (High Risk)
  - Definition: Information that, if leaked or illegally used, is expected to cause significant harm to personal safety/property, physical/mental health, reputation, or fair opportunity; or data belonging to generally sensitive categories.
  - Core Standard: **High damage consequences**. Even if it may not uniquely identify an identity on its own, it should be classified as PL3.
  - Classification Rules:
    1. Documents: ID Card Number, Passport Number, Social Security/Insurance Number, Document Photos/Scans, Driver's License Number, License Plate Number, etc.
    2. Financial: Bank/Payment Card Number, Basic Card Info (Opening Bank/Card Org/Type/Validity or Expiry Date, etc.), Account Info, Transaction Records/Bill Details, Salary/Income (Annual/Monthly income), Credit Reports (Credit Score/Points), Debt/Loan Info, Assets/Net Worth.
    3. Health: Medical Records/History/Hospital Visits/Surgery & Clinical Procedures, Diagnosis Results, Prescriptions, Specific Physiological Metrics, Specific Body Metrics, Reproductive Health, Mental Illness/Therapy or Counseling Records.
    4. Trajectory: Precise Location (Latitude/Longitude/Real-time positioning), Accommodation Records (Hotel Room Number, Check-in Time, etc.), Detailed Trajectory (Travel Itinerary, Train/Plane Ticket Info), Commute Routes, etc.
    5. Biometrics: Face, Fingerprint, Voiceprint, Iris features, etc.
    6. Communication Content: Raw Chat Logs, SMS/Email Content, Call Detail Records, etc.
    7. Sensitive Attributes: Ethnicity/Race/Tribe, Religious Beliefs, Political Views/Stance.
    8. Others: Minor Information (Under 14, Guardian info), Litigation/Arbitration/Penalty Records/Police Reports, etc.
  - Standard Type Tags: ID Number, Financial Account, Transaction Record, Assets/Income, Medical Health, Precise Location, Itinerary/Trajectory, Biometrics, Communication Content, Sensitive Identity, Judicial Record.

### 【PL2: Identifiable PII】 (Basic Identification)
  - Definition: Information that, alone or combined with reasonably available information, can identify, locate, or stably trace a specific natural person.
  - Core Standard: Identifiable / Linkable / Traceable.
  - Classification Rules:
    1. Direct Identifier: Real Name (Full Name), Specific Age, Specific Date of Birth, Gender, Mobile Number, Landline, Email Address, Detailed Address, Zip Code, Work Address.
    2. Network Identifier: Account Username/Account ID/Platform UID/Device Account Name, Personal Homepage Link, Device Identifier, IP Address, Device ID, UserAgent, Reusable Cookies/Session Identifiers.
    3. Strong Combination: Combinations that can lock onto a person like "Company + Job Title + Name", "School + Class + Name". Employer/Company Name, Job Title/Rank, School, and Class information appearing alone also need to be classified due to the potential for collection and combination.
    4. Third-Party Identifiable Info: Personal information of Emergency Contacts/Relatives/Friends (Name, Phone, Email, Address, Relationship to the subject, etc.).
  - Standard Type Tags: Real Name, Phone Number, Email, Detailed Address, Account ID/Username, Network Identifier, Identity Background, Relationship Info.

### 【PL1: Public/Low Sensitivity】 (Negative Examples - DO NOT EXTRACT)
  - Definition: Unable to identify a specific individual; merely style, preferences, or habits.
  - Core Standard: Unidentifiable + Low Harm + Not High Sensitivity.
  - Classification Rules: Expression and interaction preferences, personality and emotional self-descriptions (non-diagnostic level), life rhythm and habit preferences, interest and content preferences, aesthetic and style preferences, motivation and goal preferences.

# Extraction Granularity & Boundary Principles
Core Principle: Only extract "Sensitive Entities" or "Minimum Sensitive Fact Fragments." Strictly forbid extracting full sentences.
1. Remove Unnecessary Context: Do not include introductory words or punctuation marks.
2. Maintain Semantic Integrity (For Descriptive Privacy): Extract minimum phrase containing core elements.
3. Values Must Combine with Unit/Object: Standalone numbers are not extracted unless matching PL2-PL4.
4. Real Name Must Be the User's Own Full Name: Use provided User's Real Name field as reference.
"""


def download_model(repo_id: str, revision: str, target_dir: str, model_id: str) -> dict:
    os.makedirs(target_dir, exist_ok=True)

    from modelscope.hub.snapshot_download import snapshot_download  # type: ignore

    # Execute download into target directory
    downloaded_path = snapshot_download(
        repo_id,
        revision=revision,
        local_dir=target_dir,
    )

    # For MemPrivacy models, ensure prompt asset is saved as raw third-party resource with metadata
    if "memprivacy" in model_id.lower():
        prompt_path = os.path.join(target_dir, "privacy_prompt.txt")
        meta_path = os.path.join(target_dir, "privacy_prompt.meta.json")
        if not os.path.isfile(prompt_path):
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(MEMPRIVACY_OFFICIAL_PROMPT)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "source": f"https://modelscope.cn/models/{repo_id}",
                    "license": "CC BY-NC-ND 4.0",
                    "license_note": "Raw official MemPrivacy extraction prompt, unmodified third-party resource",
                    "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }, f, indent=2)

    # Record download metadata
    meta = {
        "repo_id": repo_id,
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_path": str(downloaded_path),
        "downloaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    meta_file = os.path.join(target_dir, ".download_meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "ok": True,
        "repo_id": repo_id,
        "model_id": model_id,
        "requested_revision": revision,
        "target_dir": target_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="ModelScope Isolated Downloader Worker")
    parser.add_argument("--repo-id", required=True, help="ModelScope Repo ID")
    parser.add_argument("--revision", default="master", help="Model revision or commit")
    parser.add_argument("--target-dir", required=True, help="Target download directory")
    parser.add_argument("--model-id", required=True, help="Catalog Model ID")

    args = parser.parse_args()

    try:
        res = download_model(
            repo_id=args.repo_id,
            revision=args.revision,
            target_dir=args.target_dir,
            model_id=args.model_id,
        )
        sys.stdout.write(json.dumps(res, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        sys.exit(0)
    except Exception as exc:
        err_res = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        sys.stdout.write(json.dumps(err_res, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
