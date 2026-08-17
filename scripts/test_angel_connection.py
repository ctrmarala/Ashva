"""
Ashva Angel One SmartAPI Authentication & Connection Tester
Tests SmartConnect session generation, TOTP verification, and outbound IP whitelisting.

Usage:
    python scripts/test_angel_connection.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.execution.angel_broker import AngelBrokerGateway


def main():
    print("=" * 80)
    print("[*] ASHVA ANGEL ONE SMARTAPI LIVE CONNECTION TESTER")
    print("=" * 80)

    config_path = "config/angel_one.yaml"
    gateway = AngelBrokerGateway.from_config_file(config_path)

    print(f"[*] Target API Key        : {gateway.api_key}")
    print(f"[*] Registered Static IP  : {gateway.whitelisted_static_ip}")
    print(f"[*] Client Code           : {gateway.client_code}")

    # 1. Check IP Compliance
    print("\n[*] Step 1: Checking outbound IPv4 compliance against Azure Static IP...")
    current_ip, is_compliant = gateway.check_outbound_ip_compliance()
    print(f"    - Current Outbound IP : {current_ip}")
    print(f"    - Matches Config IP?  : {'[YES] MATCHED [PASS]' if is_compliant else '[NO] MISMATCH [WARN]'}")

    if not is_compliant:
        print("\n[NOTE] If calling directly from laptop, make sure either:")
        print(f"  a) You entered your laptop IP ({current_ip}) into 'Secondary Static IP' in Angel One portal, OR")
        print(f"  b) You route traffic through your Azure VM proxy ({gateway.whitelisted_static_ip}).")

    # 2. Test SmartAPI Session Login if Client Code is provided
    if gateway.client_code and gateway.client_code != "YOUR_CLIENT_CODE":
        print("\n[*] Step 2: Attempting SmartConnect Session Handshake with TOTP...")
        try:
            session_data = gateway.authenticate()
            print("[+] Session Authentication: SUCCESSFUL [PASS]")
            print(f"    - Feed Token : {session_data.get('feedToken')[:15]}...")
            print(f"    - User Name  : {session_data.get('userName', 'N/A')}")
        except Exception as e:
            print(f"[!] Authentication Error: {e}")
    else:
        print("\n[*] Step 2: Ready for Login Credentials.")
        print("  To test actual login, please add your Client Code, Password/PIN, and TOTP Secret in config/angel_one.yaml")

    print("=" * 80)


if __name__ == "__main__":
    main()
