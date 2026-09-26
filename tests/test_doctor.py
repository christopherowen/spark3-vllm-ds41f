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


HEALTHY_GIDS = """rocep1s0f0 0 IB/RoCEv1 fe80:0000:0000:0000:4ebb:47ff:fee9:7f2b enp1s0f0np0
rocep1s0f0 1 RoCEv2 fe80:0000:0000:0000:4ebb:47ff:fee9:7f2b enp1s0f0np0
rocep1s0f0 2 IB/RoCEv1 0000:0000:0000:0000:0000:ffff:c0a8:0202 enp1s0f0np0
rocep1s0f0 3 RoCEv2 0000:0000:0000:0000:0000:ffff:c0a8:0202 enp1s0f0np0
"""

# dgx3 after its peers rebooted while a starting service held RDMA resources.
SHIFTED_GIDS = """rocep1s0f0 0 IB/RoCEv1 fe80:0000:0000:0000:4ebb:47ff:fee9:7f2b enp1s0f0np0
rocep1s0f0 1 RoCEv2 fe80:0000:0000:0000:4ebb:47ff:fee9:7f2b enp1s0f0np0
rocep1s0f0 2 IB/RoCEv1 0000:0000:0000:0000:0000:ffff:c0a8:0202 enp1s0f0np0
rocep1s0f0 4 RoCEv2 0000:0000:0000:0000:0000:ffff:c0a8:0202 enp1s0f0np0
"""


class RoceGidTest(unittest.TestCase):
    def test_ipv4_roce_v2_at_the_configured_index_passes(self) -> None:
        self.assertEqual(spark3.roce_gid_problems("dgx1", 3, ["rocep1s0f0"], HEALTHY_GIDS), [])

    def test_shifted_gid_names_the_slot_and_interface(self) -> None:
        problems = spark3.roce_gid_problems("dgx3", 3, ["rocep1s0f0"], SHIFTED_GIDS)
        self.assertEqual(len(problems), 1)
        self.assertIn("rocep1s0f0 GID index 3 is empty", problems[0])
        self.assertIn("index 4; re-activate enp1s0f0np0", problems[0])

    def test_ipv6_roce_v2_at_the_index_is_reported(self) -> None:
        problems = spark3.roce_gid_problems("dgx1", 1, ["rocep1s0f0"], HEALTHY_GIDS)
        self.assertIn("GID index 1 is RoCEv2 fe80", problems[0])
        self.assertIn("at index 3", problems[0])

    def test_missing_device_and_missing_ipv4_are_reported(self) -> None:
        output = "rocep1s0f1 missing\nrocep1s0f0 1 RoCEv2 fe80:0000:0000:0000:4ebb:47ff:fee9:7f2b enp1s0f0np0\n"
        problems = spark3.roce_gid_problems("dgx2", 3, ["rocep1s0f0", "rocep1s0f1"], output)
        self.assertEqual(
            problems,
            [
                "dgx2: rocep1s0f0 GID index 3 is empty; it has no IPv4 RoCE v2 GID",
                "dgx2: RDMA device rocep1s0f1 is missing",
            ],
        )


class ClockLatchTest(unittest.TestCase):
    def test_serving_node_at_full_clock_passes(self) -> None:
        self.assertEqual(spark3.clock_latch_problems("dgx1", facts(gpu="2411, 11.47"), True), [])

    def test_latched_serving_node_is_reported(self) -> None:
        # dgx3 on 2026-09-26: 520-565 MHz at about 10 W, ignoring nvidia-smi -lgc.
        problems = spark3.clock_latch_problems("dgx3", facts(gpu="559, 9.90"), True)
        self.assertEqual(len(problems), 1)
        self.assertIn("GPU clock is 559 MHz at 9.90 W while serving", problems[0])

    def test_idle_node_without_the_service_is_not_judged(self) -> None:
        self.assertEqual(spark3.clock_latch_problems("dgx2", facts(gpu="208, 4.1"), False), [])

    def test_unreadable_clock_is_reported(self) -> None:
        self.assertEqual(
            spark3.clock_latch_problems("dgx2", facts(gpu=""), True),
            ["dgx2: cannot read the GPU clock"],
        )


if __name__ == "__main__":
    unittest.main()
