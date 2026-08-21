"""
Ashva Trading Manifest & Alpha Contract Registry
Serves as the strict boundary between Alpha Factory qualification and Trading Engine execution.
Loads, validates, and exposes active approved QualifiedAlphaContracts without parameter drift.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.trading.contract import QualifiedAlphaContract

logger = logging.getLogger("Ashva.TradingManifest")


class TradingManifest:
    """
    Registry of Active Qualified Alpha Contracts approved by the Alpha Factory.
    """

    def __init__(self, contracts: Optional[List[QualifiedAlphaContract]] = None):
        self._contracts: Dict[str, QualifiedAlphaContract] = {}
        if contracts:
            for c in contracts:
                self.register_contract(c)

    def register_contract(self, contract: QualifiedAlphaContract):
        """Registers a qualified alpha contract."""
        self._contracts[contract.alpha_id] = contract
        logger.info(
            f"Registered Alpha Contract: {contract.alpha_id} v{contract.alpha_version} "
            f"[{contract.category}] Universe: {len(contract.universe)} symbols | Status: {contract.status}"
        )

    def get_contract(self, alpha_id: str) -> Optional[QualifiedAlphaContract]:
        """Retrieves a contract by alpha_id."""
        return self._contracts.get(alpha_id)

    def get_active_contracts(self) -> List[QualifiedAlphaContract]:
        """Returns all contracts with status == 'ACTIVE'."""
        return [c for c in self._contracts.values() if c.status == "ACTIVE"]

    def get_contracts_for_symbol(self, symbol: str) -> List[QualifiedAlphaContract]:
        """Returns active contracts trading the specified symbol, sorted by priority score."""
        s_clean = symbol.upper()
        matching = [
            c for c in self.get_active_contracts()
            if s_clean in [sym.upper() for sym in c.universe]
        ]
        matching.sort(key=lambda c: c.priority_score, reverse=True)
        return matching

    def get_all_symbols(self) -> List[str]:
        """Returns union of all universe symbols across active contracts."""
        sym_set = set()
        for c in self.get_active_contracts():
            for s in c.universe:
                sym_set.add(s.upper())
        return sorted(list(sym_set))

    def set_contract_status(self, alpha_id: str, status: str):
        """Allows enabling/disabling an alpha via manual controls ('ACTIVE', 'PAUSED', 'RETIRED')."""
        import dataclasses
        if alpha_id in self._contracts:
            old_c = self._contracts[alpha_id]
            updated_c = dataclasses.replace(old_c, status=status)
            self._contracts[alpha_id] = updated_c
            logger.info(f"Alpha {alpha_id} status updated to {status}")

    def to_manifest_dict(self) -> Dict[str, Any]:
        """Exports manifest metadata for ledger logging and auditability."""
        return {
            "timestamp": datetime.now().isoformat(),
            "active_alpha_count": len(self.get_active_contracts()),
            "total_registered_count": len(self._contracts),
            "contracts": [c.to_dict() for c in self._contracts.values()],
        }
