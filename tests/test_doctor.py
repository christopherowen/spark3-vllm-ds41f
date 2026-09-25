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


FAN_WORKING = (
    "modeset=Y\n"
    "kernel=7.0.0-1019-nvidia\n"
    "fan_dkms=dgx-spark-fan-control/0.1.3, 7.0.0-1019-nvidia, aarch64: installed\n"
    "fan_module=1\n"
    "fan_cooling_device=1\n"
    "fan_service=enabled/active\n"
)


def facts(**changes: str) -> dict[str, str]:
    values = spark3.host_facts(FAN_WORKING)
    values.update(changes)
    return values


class ModesetTest(unittest.TestCase):
    def test_kernel_mode_setting_passes(self) -> None:
        self.assertEqual(spark3.modeset_problems("dgx1", facts()), [])

    def test_disabled_mode_setting_is_reported(self) -> None:
        problems = spark3.modeset_problems("dgx1", facts(modeset="N"))
        self.assertEqual(len(problems), 1)
        self.assertIn("modeset is N, expected Y", problems[0])

    def test_unreadable_mode_setting_is_reported(self) -> None:
        problems = spark3.modeset_problems("dgx1", facts(modeset=""))
        self.assertIn("modeset is unreadable", problems[0])


class FanControlTest(unittest.TestCase):
    def test_working_fan_floor_passes(self) -> None:
        self.assertEqual(spark3.fan_control_problems("dgx1", facts()), [])

    def test_missing_dkms_module_is_the_only_report(self) -> None:
        problems = spark3.fan_control_problems(
            "dgx3", facts(fan_dkms="", fan_module="0", fan_cooling_device="0", fan_service="/inactive")
        )
        self.assertEqual(
            problems, ["dgx3: DKMS dgx-spark-fan-control is not installed for 7.0.0-1019-nvidia"]
        )

    def test_unloaded_module_is_reported(self) -> None:
        problems = spark3.fan_control_problems("dgx3", facts(fan_module="0"))
        self.assertIn("dgx_ec_fan_control is not loaded", problems[0])

    def test_refused_cooling_device_is_reported(self) -> None:
        # dgx3 on firmware 5.36_0ACUM027: the driver loads, then the EC rejects
        # its capability read and it refuses to register the cooling device.
        problems = spark3.fan_control_problems(
            "dgx3", facts(fan_cooling_device="0", fan_service="disabled/inactive")
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("cooling device is missing", problems[0])

    def test_stopped_daemon_is_reported(self) -> None:
        problems = spark3.fan_control_problems("dgx1", facts(fan_service="enabled/failed"))
        self.assertEqual(
            problems, ["dgx1: dgx-fan-control.service is enabled/failed, expected enabled/active"]
        )


if __name__ == "__main__":
    unittest.main()
