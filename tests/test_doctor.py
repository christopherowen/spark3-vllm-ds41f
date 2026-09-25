"""Host-independent tests for doctor's node checks."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark3", str(ROOT / "bin" / "spark3"))
spec = importlib.util.spec_from_loader("spark3", loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)


class DesktopTest(unittest.TestCase):
    def test_headless_node_passes(self) -> None:
        self.assertEqual(spark3.desktop_problems("dgx1", "multi-user.target\ninactive\n"), [])

    def test_graphical_target_and_running_display_manager_are_reported(self) -> None:
        problems = spark3.desktop_problems("dgx2", "graphical.target\nactive\n")
        self.assertEqual(len(problems), 2)
        self.assertIn("boots to graphical.target", problems[0])
        self.assertIn("display manager is active", problems[1])

    def test_display_manager_started_by_hand_is_reported(self) -> None:
        problems = spark3.desktop_problems("dgx3", "multi-user.target\nactive\n")
        self.assertEqual(problems, ["dgx3: display manager is active"])

    def test_missing_output_is_reported(self) -> None:
        problems = spark3.desktop_problems("dgx1", "")
        self.assertEqual(len(problems), 1)
        self.assertIn("unknown target", problems[0])


DGX1_BOOT = """BootCurrent: 0001
Timeout: 1 seconds
BootOrder: 0001,0003,0004,0005,0006
Boot0001* ubuntu\tHD(1,GPT,f9e7b4d5-76a3-46f4-8fbb-cea82cd05534,0x800,0x95000)/File(\\EFI\\ubuntu\\shimaa64.efi) File(.)
Boot0003* UEFI: PXE IPv4 Realtek PCIe 10 GBE Family Controller\tPcieRoot(0x7)/Pci(0x0,0x0)/Pci(0x0,0x0)/MAC(4cbb47e97f2a,0)
Boot0004* UEFI:CD/DVD Drive\tBBS(129,,0x0)
"""

# dgx3 after its 2026-06-12 firmware update, before the boot order was fixed.
DGX3_PXE_FIRST_BOOT = """BootCurrent: 0004
Timeout: 1 seconds
BootOrder: 0002,0004
Boot0002* UEFI: PXE IPv4 Realtek PCIe 10 GBE Family Controller\tPcieRoot(0x7)/Pci(0x0,0x0)/Pci(0x0,0x0)/MAC(4cbb47e97f2a,0)
Boot0004* ubuntu\tHD(1,GPT,88df73c8-497a-4b1d-be95-4e318905d1fb,0x800,0x95000)/File(\\EFI\\ubuntu\\shimaa64.efi) File(.)
"""


class BootOrderTest(unittest.TestCase):
    def test_installed_os_first_passes(self) -> None:
        self.assertEqual(spark3.boot_order_problems("dgx1", DGX1_BOOT), [])

    def test_network_boot_first_is_reported(self) -> None:
        problems = spark3.boot_order_problems("dgx3", DGX3_PXE_FIRST_BOOT)
        self.assertEqual(len(problems), 1)
        self.assertIn("UEFI: PXE IPv4 Realtek PCIe 10 GBE Family Controller", problems[0])
        self.assertIn("BootOrder 0002,0004", problems[0])

    def test_fixed_order_passes(self) -> None:
        fixed = DGX3_PXE_FIRST_BOOT.replace("BootOrder: 0002,0004", "BootOrder: 0004,0002")
        self.assertEqual(spark3.boot_order_problems("dgx3", fixed), [])

    def test_unreadable_boot_order_is_reported(self) -> None:
        self.assertEqual(
            spark3.boot_order_problems("dgx2", ""),
            ["dgx2: cannot read the UEFI boot order"],
        )


if __name__ == "__main__":
    unittest.main()
