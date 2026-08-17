"""
Ashva SEBI Static IPv4 Compliance Checker for Angel One SmartAPI
Verifies that your current outbound public IPv4 address matches the static IP registered
on the Angel One SmartAPI developer portal (https://smartapi.angelbroking.com).

Usage:
    python scripts/verify_angel_ip.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.execution.angel_broker import AngelBrokerGateway


def main():
    print("=" * 80)
    print("[*] ASHVA SEBI STATIC IP & ANGEL ONE COMPLIANCE VERIFIER")
    print("=" * 80)

    # 1. Initialize Gateway
    config_path = Path("config/angel_one.yaml")
    if not config_path.exists():
        print("[!] config/angel_one.yaml not found. Checking current machine public IPv4 directly...")
        gateway = AngelBrokerGateway(whitelisted_static_ip=None)
    else:
        gateway = AngelBrokerGateway.from_config_file(str(config_path))

    # 2. Check Outbound IPv4
    print("[*] Querying outbound IPv4 address from network interface...")
    current_ip, is_compliant = gateway.check_outbound_ip_compliance()

    print(f"\n[+] Detected Outbound IPv4 Address : {current_ip}")
    
    if gateway.whitelisted_static_ip:
        print(f"[+] Whitelisted Static IP (Config): {gateway.whitelisted_static_ip}")
        if is_compliant:
            print("\n[SUCCESS] SEBI STATIC IP COMPLIANCE: VERIFIED ✅")
            print("  Your outbound IP matches the registered static IP. Angel One SmartAPI will accept orders.")
        else:
            print("\n[WARNING] SEBI STATIC IP MISMATCH ⚠️")
            print(f"  Your current IP ({current_ip}) does NOT match config ({gateway.whitelisted_static_ip}).")
            print("  Please register this IP on https://smartapi.angelbroking.com or configure a static proxy.")
    else:
        print("\n[ACTION REQUIRED] Register This Static IPv4 on Angel One:")
        print(f"  1. Log in to https://smartapi.angelbroking.com")
        print(f"  2. Edit your Trading App settings.")
        print(f"  3. Set 'Static IP' to: {current_ip}")
        print(f"  4. Add 'whitelisted_static_ip: \"{current_ip}\"' to config/angel_one.yaml")

    print("=" * 80)


if __name__ == "__main__":
    main()
