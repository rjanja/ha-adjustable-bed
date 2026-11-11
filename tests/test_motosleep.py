"""Tests for MotoSleep, Power Bob, and binary MOTO model routing."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from custom_components.adjustable_bed.beds.motosleep import (
    INTELLIBED_FULL_TRAVEL,
    MotoSleepController,
    build_moto_binary_frame,
    build_motosleep_numeric_frame,
    build_motosleep_sync_frame,
    build_power_bob_numeric_frame,
    parse_intellibed_status,
)
from custom_components.adjustable_bed.beds.motosleep_profiles import (
    MotoSleepTransport,
    resolve_motosleep_profile,
)
from custom_components.adjustable_bed.const import (
    BED_TYPE_INTELLIBED,
    BED_TYPE_MOTOSLEEP,
    MOTOSLEEP_CHAR_UUID,
    bed_type_has_position_feedback,
)
from custom_components.adjustable_bed.controller_factory import create_controller
from custom_components.adjustable_bed.cover import _build_cover_description
from custom_components.adjustable_bed.detection import detect_bed_type


def _moto_name(page: str, *, feature: str = "Q", led: str = "D") -> str:
    """Build an exact-length name accepted by MotoPageModel."""
    return f"MOTOB00{page}4{led}{feature}NX" + "0" * 15


@pytest.mark.parametrize(
    ("name", "profile_id"),
    [
        ("HHC0000015CDEF", "power_bob_a15"),
        ("HHC0000035CDEF", "power_bob_d345"),
        ("HHC0000040CDEF", "power_bob_four"),
        ("HHC0000095CDEF", "power_bob_wr1140"),
        ("HHC0000000CDEF", "power_bob_zero"),
        ("HHC0000010CDEF", "power_bob_one"),
        ("HHC0000020CDEF", "power_bob_two"),
        ("HHC0000030CDEF", "power_bob_three"),
        ("HHC0000050CDEF", "power_bob_five"),
        ("HHC0000060CDEF", "power_bob_six"),
        ("HHC0000070CDEF", "power_bob_seven"),
        ("HHC0000080CDEF", "power_bob_eight"),
        ("HHC0000090CDEF", "power_bob_nine"),
        ("HHC00000D0CDEF", "power_bob_six_zg"),
    ],
)
def test_power_bob_selector_matrix(name: str, profile_id: str) -> None:
    """Every root selector from Power Bob 2.0.3 resolves independently."""
    assert len(name) == 14
    assert resolve_motosleep_profile(name).profile_id == profile_id
    assert resolve_motosleep_profile(name.lower()).profile_id == profile_id


def test_issue_445_name_routes_pq_to_lumbar() -> None:
    """The reported A15 name uses P/Q, not p/q, for manual lumbar."""
    profile = resolve_motosleep_profile("HHC0069815CDEF")

    assert profile.profile_id == "power_bob_a15"
    assert profile.transport is MotoSleepTransport.POWER_BOB_ASCII
    assert profile.motors["lumbar"] == ("P", "Q")
    assert "auxiliary" not in profile.motors


async def test_issue_468_short_panel_eight_keeps_home_preset() -> None:
    """The reported short HHC PanelEight bed retains its working Home action."""
    controller, client = _controller("HHC0120182CDEH")

    assert controller.profile.profile_id == "power_bob_eight"
    assert controller.supports_preset_flat is True

    await controller.preset_flat()

    client.write_gatt_char.assert_awaited_once_with(MOTOSLEEP_CHAR_UUID, b"$O", response=False)


async def test_issue_470_short_panel_eight_keeps_toggle_only_light() -> None:
    """The reported PanelEight bed uses its root light toggle, not generic RGB."""
    controller, client = _controller("HHC0120182CDEH")

    assert controller.profile.profile_id == "power_bob_eight"
    assert controller.profile.light_toggle == "A"
    assert controller.profile.rgb_light is False
    assert controller.supports_lights is True
    assert controller.supports_light_toggle_control is True
    assert controller.supports_light_color_control is False
    assert controller.supports_discrete_light_control is False

    await controller.lights_toggle()

    client.write_gatt_char.assert_awaited_once_with(MOTOSLEEP_CHAR_UUID, b"$A", response=False)


def test_power_bob_accepts_exact_length_name_containing_hhc() -> None:
    """Power Bob filters for a case-sensitive HHC substring, not a prefix."""
    profile = resolve_motosleep_profile("XXHHC000150000")

    assert len("XXHHC000150000") == 14
    assert profile.profile_id == "power_bob_a15"


def test_power_bob_unmatched_selector_exposes_no_guessed_panel() -> None:
    """The APK has no default root route for unmatched 14-character selectors."""
    profile = resolve_motosleep_profile("XXHHC000X00000")

    assert len("XXHHC000X00000") == 14
    assert profile.profile_id == "power_bob_unknown"
    assert not profile.motors
    assert profile.stop is None


def test_power_bob_capabilities_are_not_universal() -> None:
    """A two-button PanelFive must not inherit lumbar, memory, or lights."""
    profile = resolve_motosleep_profile("HHC0000050CDEF")

    assert profile.motors == {"back": ("K", "L")}
    assert not profile.presets
    assert not profile.memory_recall
    assert profile.light_toggle is None
    assert profile.stop is None


@pytest.mark.parametrize("name", ["HHC0000010CDEF", "HHC0000050CDEF"])
def test_power_bob_minimal_panels_keep_orthogonal_rgb_settings(name: str) -> None:
    """Mood/Night PanelRGB routing is independent of the root motor panel."""
    profile = resolve_motosleep_profile(name)
    controller, _client = _controller(name)

    assert profile.profile_id in {"power_bob_one", "power_bob_five"}
    assert profile.rgb_light is True
    assert profile.light_toggle is None
    assert controller.supports_light_color_control is True
    assert controller.supports_light_toggle_control is False


def test_neutral_axis_builds_a_generic_cover_description() -> None:
    """An APK-proven axis with tentative semantics must still create an entity."""
    controller, _client = _controller("HHC0000040CDEF")
    auxiliary = next(spec for spec in controller.motor_control_specs if spec.key == "auxiliary_1")

    description = _build_cover_description(auxiliary)

    assert description.key == "auxiliary_1"
    assert description.translation_key == "auxiliary_1"
    assert description.icon == "mdi:bed-outline"


def _hhc_name(c16: str, c18: str = "0", c22: str = "0", c26: str = "0") -> str:
    chars = list("HHC" + "0" * 24)
    chars[16] = c16
    chars[18] = c18
    chars[22] = c22
    chars[26] = c26
    return "".join(chars)


@pytest.mark.parametrize(
    ("name", "profile_id"),
    [
        (_hhc_name("F"), "motosleep_hhc_f"),
        (_hhc_name("G"), "motosleep_hhc_g"),
        (_hhc_name("H"), "motosleep_hhc_h"),
        (_hhc_name("1", "5"), "motosleep_hhc_a"),
        (_hhc_name("4", "6"), "motosleep_hhc_one"),
        (_hhc_name("3", "5"), "motosleep_hhc_d"),
        (_hhc_name("4", "0"), "motosleep_hhc_four"),
        (_hhc_name("9", "5"), "motosleep_hhc_wr1140"),
        (_hhc_name("0"), "motosleep_hhc_zero"),
        (_hhc_name("2"), "motosleep_hhc_two"),
        (_hhc_name("3"), "motosleep_hhc_three"),
        (_hhc_name("5", c26="I"), "motosleep_hhc_five_new"),
        (_hhc_name("5"), "motosleep_hhc_five"),
        (_hhc_name("6", c26="I"), "motosleep_hhc_six_new"),
        (_hhc_name("6"), "motosleep_hhc_six"),
        (_hhc_name("7"), "motosleep_hhc_seven"),
        (_hhc_name("8", c26="I"), "motosleep_hhc_eight_new"),
        (_hhc_name("8"), "motosleep_hhc_eight"),
        (_hhc_name("9"), "motosleep_hhc_nine"),
        (_hhc_name("D"), "motosleep_hhc_six_zg"),
        (_hhc_name("E"), "motosleep_hhc_e"),
    ],
)
def test_motosleep_hhc_selector_matrix(name: str, profile_id: str) -> None:
    """Every MotoSleep HHC page selector resolves to its own profile."""
    assert resolve_motosleep_profile(name).profile_id == profile_id
    assert resolve_motosleep_profile(name.lower()).profile_id == profile_id


def test_hhc_raw_and_wrapped_transport_selection() -> None:
    """Name index 22 selects the APK's raw HHC write path."""
    raw = resolve_motosleep_profile(_hhc_name("G"))
    wrapped = resolve_motosleep_profile(_hhc_name("G", c22="M"))

    assert raw.raw_hhc is True
    assert wrapped.raw_hhc is False


def test_panel_e_preserves_unlabelled_w_action() -> None:
    """PanelE's proven $W callsite remains available under a neutral label."""
    profile = resolve_motosleep_profile(_hhc_name("E"))

    assert profile.auxiliary_action == "W"


def test_current_hhc_accepts_long_name_containing_hhc() -> None:
    """MotoSleep's scan filter also accepts HHC away from index zero."""
    chars = list(_hhc_name("G"))
    chars[:6] = "XXXHHC"

    assert resolve_motosleep_profile("".join(chars)).profile_id == "motosleep_hhc_g"


@pytest.mark.parametrize(
    ("name", "expected_motors"),
    [
        (
            _hhc_name("1", "5", c26="H"),
            {"back": ("K", "L"), "legs": ("M", "N"), "lumbar": ("p", "q")},
        ),
        (_hhc_name("1", "5"), {"back": ("K", "L"), "legs": ("M", "N"), "tilt": ("P", "Q")}),
        (
            _hhc_name("3", "5", c26="H"),
            {"back": ("K", "L"), "legs": ("M", "N"), "neck": ("p", "q")},
        ),
        (_hhc_name("3", "6"), {"back": ("K", "L"), "legs": ("M", "N"), "tilt": ("P", "Q")}),
        (
            _hhc_name("4", "5", c26="H"),
            {"back": ("K", "L"), "legs": ("M", "N"), "lumbar": ("p", "q")},
        ),
        (_hhc_name("8", "5"), {"back": ("K", "L"), "legs": ("M", "N"), "tilt": ("P", "Q")}),
        (_hhc_name("C"), {"back": ("K", "L"), "legs": ("M", "N"), "auxiliary": ("P", "Q")}),
    ],
)
def test_hhc_conditional_axis_labels(
    name: str, expected_motors: dict[str, tuple[str, str]]
) -> None:
    """Only axes with an APK-proven physical label receive that label."""
    assert resolve_motosleep_profile(name).motors == expected_motors


@pytest.mark.parametrize(
    ("page", "profile_id"),
    [
        ("0", "motosleep_moto_wrs23ms"),
        ("1", "motosleep_moto_wrs14dmm"),
        ("2", "motosleep_moto_wrs18ms"),
        ("3", "motosleep_moto_wrs14ms"),
        ("4", "motosleep_moto_wrs20ms"),
        ("5", "motosleep_moto_wrs16ms"),
        ("6", "motosleep_moto_wrc30mms"),
        ("7", "motosleep_moto_wrs27ms"),
        ("8", "motosleep_moto_wrs20ms_swing"),
        ("9", "motosleep_moto_wr219"),
    ],
)
def test_binary_moto_selector_matrix(page: str, profile_id: str) -> None:
    """All ten binary bed pages resolve from local-name index 7."""
    name = _moto_name(page)

    assert len(name) == 28
    assert resolve_motosleep_profile(name).profile_id == profile_id


@pytest.mark.parametrize(
    ("page", "expected_motors"),
    [
        (
            "0",
            {
                "back": (0x0002, 0x0001),
                "legs": (0x0200, 0x0100),
                "head": (0x0020, 0x0010),
                "lumbar": (0x0080, 0x0040),
                "feet": (0x0008, 0x0004),
            },
        ),
        ("1", {"back": (0x0020, 0x0010), "legs": (0x0200, 0x0100), "head": (0x0002, 0x0001)}),
        (
            "2",
            {
                "back": (0x0002, 0x0001),
                "legs": (0x0200, 0x0100),
                "lumbar": (0x0080, 0x0040),
                "feet": (0x0008, 0x0004),
            },
        ),
        ("3", {"back": (0x0002, 0x0001), "legs": (0x0200, 0x0100), "feet": (0x0008, 0x0004)}),
        (
            "4",
            {
                "back": (0x0002, 0x0001),
                "legs": (0x0200, 0x0100),
                "head": (0x0020, 0x0010),
                "lumbar": (0x0080, 0x0040),
                "feet": (0x0008, 0x0004),
            },
        ),
        ("5", {"back": (0x0002, 0x0001), "legs": (0x0200, 0x0100), "feet": (0x0008, 0x0004)}),
        (
            "6",
            {
                "neck": (0x0020, 0x0010),
                "back": (0x0002, 0x0001),
                "lumbar": (0x0080, 0x0040),
                "legs": (0x0008, 0x0004),
            },
        ),
        ("7", {"back": (0x0002, 0x0001), "legs": (0x0008, 0x0004), "lumbar": (0x0080, 0x0040)}),
        (
            "8",
            {
                "back": (0x0002, 0x0001),
                "legs": (0x0200, 0x0100),
                "head": (0x0020, 0x0010),
                "lumbar": (0x0080, 0x0040),
                "feet": (0x0008, 0x0004),
            },
        ),
        ("9", {"back": (0x000A, 0x0005), "lumbar": (0x0200, 0x0100), "legs": (0x00A0, 0x0050)}),
    ],
)
def test_binary_moto_motor_matrix(page: str, expected_motors: dict[str, tuple[int, int]]) -> None:
    """Every binary page exposes only its app-rendered actuator matrix."""
    assert resolve_motosleep_profile(_moto_name(page)).motors == expected_motors


def test_binary_moto_unknown_name_is_conservative() -> None:
    """MOTO-like audio/amplifier names do not inherit bed controls."""
    profile = resolve_motosleep_profile("MOTOAMP")

    assert profile.profile_id == "motosleep_moto_unknown"
    assert not profile.motors
    assert profile.stop is None


def test_binary_moto_unknown_page_is_conservative() -> None:
    """A regex-shaped name with an unknown page cannot crash profile routing."""
    profile = resolve_motosleep_profile(_moto_name("X"))

    assert profile.profile_id == "motosleep_moto_unknown"
    assert profile.transport is MotoSleepTransport.MOTO_BINARY
    assert not profile.motors
    assert profile.stop is None


def test_lowercase_moto_name_routes_to_the_exact_binary_profile() -> None:
    """Case-insensitive detection and profile routing select the same model."""
    profile = resolve_motosleep_profile(_moto_name("4").lower())

    assert profile.profile_id == "motosleep_moto_wrs20ms"
    assert profile.transport is MotoSleepTransport.MOTO_BINARY
    assert profile.motors["lumbar"] == (0x0080, 0x0040)
    assert profile.stop == ((0x0000, 0), 5, 100)


@pytest.mark.parametrize(
    ("command", "data", "expected"),
    [
        (0x0000, 0, "24230000000000410d"),
        (0x0002, 0, "24230002000002410d"),
        (0x8009, 0, "24238009000089410d"),
        (0x9088, 1, "24239088000119410d"),
    ],
)
def test_binary_frame_matches_clean_room_vectors(command: int, data: int, expected: str) -> None:
    """Binary framing and XOR match the frozen MotoSleep APK vectors."""
    assert build_moto_binary_frame(command, data).hex() == expected


def test_numeric_frame_order_differs_between_apps() -> None:
    """Power Bob and MotoSleep place selector/value fields in opposite order."""
    assert build_power_bob_numeric_frame("00319", "00128") == b"$#003190012800000R\r"
    assert build_motosleep_numeric_frame("00128", "00319") == b"$#001280031900000R\r"
    assert build_motosleep_sync_frame("00015") == b"$#000150000000004R\r"


def _controller(name: str) -> tuple[MotoSleepController, MagicMock]:
    client = MagicMock()
    client.is_connected = True
    client.services = []
    client.write_gatt_char = AsyncMock()
    coordinator = MagicMock()
    coordinator.address = "AA:BB:CC:DD:EE:FF"
    coordinator.client = client
    coordinator.cancel_command = asyncio.Event()
    coordinator.motor_pulse_count = 1
    coordinator.record_command_trace = MagicMock()
    return MotoSleepController(coordinator, device_name=name), client


async def test_issue_445_manual_lumbar_writes_raw_p_uppercase() -> None:
    """A15 manual lumbar uses the Power Bob no-response P command."""
    controller, client = _controller("HHC0069815CDEF")

    with patch("custom_components.adjustable_bed.beds.motosleep.asyncio.sleep") as sleep:
        await controller.move_lumbar_up()

    sleep.assert_awaited_once_with(0.1)
    client.write_gatt_char.assert_awaited_once_with(MOTOSLEEP_CHAR_UUID, b"$P", response=False)


async def test_factory_passes_device_name_into_profile_routing(hass) -> None:
    """Controller creation retains the advertised name used by both APK routers."""
    coordinator = MagicMock(hass=hass)
    controller = await create_controller(
        coordinator,
        BED_TYPE_MOTOSLEEP,
        protocol_variant=None,
        client=None,
        device_name="HHC0069815CDEF",
    )

    assert isinstance(controller, MotoSleepController)
    assert controller.profile.profile_id == "power_bob_a15"


async def test_current_hhc_wraps_short_actions_and_stops_five_times() -> None:
    """MotoSleep HHC writes wrapped actions and the recovered release schedule."""
    controller, client = _controller(_hhc_name("G", c22="M"))

    await controller.move_lumbar_up()

    calls = client.write_gatt_char.await_args_list
    assert calls[0].args[1] == b"$#$pR\r"
    assert [call.args[1] for call in calls[1:]] == [b"$#$bR\r"] * 5
    assert all(call.kwargs["response"] is True for call in calls)


async def test_panel_e_auxiliary_action_writes_exact_frame() -> None:
    """The neutral PanelE action follows its model's exact raw transport."""
    controller, client = _controller(_hhc_name("E"))

    await controller.auxiliary_action()

    client.write_gatt_char.assert_awaited_once_with(MOTOSLEEP_CHAR_UUID, b"$W", response=True)


def test_control_characteristic_prefers_ffe1_and_falls_back_to_fff1() -> None:
    """The controller follows both APKs' discovered write-role precedence."""
    controller, client = _controller(_hhc_name("G"))
    fff1 = MagicMock(uuid="0000fff1-0000-1000-8000-00805f9b34fb")
    client.services = [MagicMock(characteristics=[fff1])]
    assert controller.control_characteristic_uuid == fff1.uuid

    ffe1 = MagicMock(uuid="0000ffe1-0000-1000-8000-00805f9b34fb")
    client.services = [MagicMock(characteristics=[fff1, ffe1])]
    assert controller.control_characteristic_uuid == ffe1.uuid


async def test_power_bob_presets_and_memory_are_single_raw_no_response_writes() -> None:
    """Power Bob action buttons use their exact raw one-shot transport."""
    controller, client = _controller("HHC0069815CDEF")

    await controller.preset_flat()
    await controller.preset_memory(2)
    await controller.program_memory(1)

    assert [call.args[1] for call in client.write_gatt_char.await_args_list] == [
        b"$O",
        b"$V",
        b"$Z",
    ]
    assert all(call.kwargs["response"] is False for call in client.write_gatt_char.await_args_list)


async def test_hhc_massage_uses_only_profile_proven_controls() -> None:
    """HHC massage capabilities and bytes come from the selected panel."""
    controller, client = _controller(_hhc_name("F", c22="M"))

    assert controller.auto_enable_massage is True
    assert controller.supports_head_massage_toggle_control is True
    assert controller.supports_head_massage_intensity_step_control is False
    await controller.massage_head_toggle()
    await controller.massage_off()

    assert [call.args[1] for call in client.write_gatt_char.await_args_list] == [
        b"$#$CR\r",
        b"$#$DR\r",
    ]


async def test_hhc_zero_profile_uses_separate_massage_intensity_actions() -> None:
    """PanelZero retains its extended head and foot massage command family."""
    controller, client = _controller(_hhc_name("0"))

    assert controller.supports_head_massage_intensity_step_control is True
    assert controller.supports_foot_massage_intensity_step_control is True
    await controller.massage_head_up()
    await controller.massage_foot_down()

    assert [call.args[1] for call in client.write_gatt_char.await_args_list] == [
        b"$G",
        b"$F",
    ]


async def test_hhc_zero_profile_routes_zone_on_off_actions_as_toggles() -> None:
    """PanelZero J/I actions retain their APK-proven on-off semantics."""
    controller, client = _controller(_hhc_name("0"))

    assert controller.profile.massage["head_toggle"] == "J"
    assert controller.profile.massage["foot_toggle"] == "I"
    assert controller.supports_head_massage_toggle_control is True
    assert controller.supports_foot_massage_toggle_control is True
    await controller.massage_head_toggle()
    await controller.massage_foot_toggle()

    assert [call.args[1] for call in client.write_gatt_char.await_args_list] == [
        b"$J",
        b"$I",
    ]


async def test_power_bob_rgb_matches_clean_room_channel_vectors() -> None:
    """Power Bob scales RGB and sends selector-before-value channel frames."""
    controller, client = _controller("HHC0069815CDEF")

    with patch("custom_components.adjustable_bed.beds.motosleep.asyncio.sleep") as sleep:
        await controller.set_light_color((255, 128, 0))

    assert [call.args[1] for call in client.write_gatt_char.await_args_list] == [
        b"$#003150012000004R\r",
        b"$#003160006000002R\r",
        b"$#003170000000005R\r",
    ]
    assert all(call.kwargs["response"] is False for call in client.write_gatt_char.await_args_list)
    assert sleep.await_args_list == [call(0.02), call(0.02)]


async def test_rgb_off_uses_absolute_zero_channels() -> None:
    """RGB profiles expose idempotent off instead of the unrelated $A toggle."""
    controller, client = _controller(_hhc_name("5", c22="M", c26="I"))

    assert controller.supports_discrete_light_control is True
    assert controller.supports_explicit_light_on_control is False
    await controller.lights_off()

    assert [call.args[1] for call in client.write_gatt_char.await_args_list] == [
        build_motosleep_numeric_frame("00000", selector) for selector in ("00315", "00316", "00317")
    ]


async def test_hhc_sync_uses_distinct_enable_and_disable_opcodes() -> None:
    """Current MainPanel bind state maps to the two proven numeric frames."""
    controller, client = _controller(_hhc_name("G", c22="M"))

    assert controller.supports_synchro is True
    await controller.set_synchro(True)
    await controller.set_synchro(False)

    assert [call.args[1] for call in client.write_gatt_char.await_args_list] == [
        build_motosleep_sync_frame("00012"),
        build_motosleep_sync_frame("00015"),
    ]


async def test_binary_preset_and_memory_frames_are_model_specific() -> None:
    """Normal MOTO pages use binary preset recall and save commands."""
    controller, client = _controller(_moto_name("4"))

    await controller.preset_tv()
    await controller.preset_memory(1)
    await controller.program_memory(2)

    assert [call.args[1] for call in client.write_gatt_char.await_args_list] == [
        build_moto_binary_frame(0x800A),
        build_moto_binary_frame(0x800C),
        build_moto_binary_frame(0x811E),
    ]


async def test_binary_standard_release_uses_five_zero_stop_frames() -> None:
    """Normal MOTO movement uses the standard five-write release helper."""
    controller, client = _controller(_moto_name("4"))

    await controller.move_back_up()

    assert [call.args[1] for call in client.write_gatt_char.await_args_list[1:]] == [
        build_moto_binary_frame(0x0000, 0)
    ] * 5


async def test_binary_wr219_uses_second_stop_vector() -> None:
    """WR219 movement ends with six 0x9088/data=1 STOP frames."""
    controller, client = _controller(_moto_name("9"))

    await controller.move_back_up()

    calls = client.write_gatt_char.await_args_list
    assert calls[0].args[1] == build_moto_binary_frame(0x000A)
    assert [call.args[1] for call in calls[1:]] == [build_moto_binary_frame(0x9088, 1)] * 6


def test_swing_is_only_exposed_by_swing_models() -> None:
    """Binary swing commands are model-specific bed functions, not audio."""
    swing = _controller(_moto_name("8"))[0]
    wr219 = _controller(_moto_name("9"))[0]
    normal = _controller(_moto_name("4"))[0]

    assert swing.supports_preset_swing is True
    assert swing.profile.swing == 0x8014
    assert wr219.profile.swing == 0x9089
    assert normal.supports_preset_swing is False


# Intellibed Sleep Genius (com.mobile.intellibed 2.1.67).  Byte vectors are the
# literals and XOR checksums of the app's own frame builders.
_INTELLIBED_NAME = "HHC14951BEDM9"


@pytest.mark.parametrize("name", ["HHC14951BEDM9", "HHC23235BEDM9", "hhc14951bedm9"])
def test_intellibed_names_route_to_their_own_profile(name: str) -> None:
    """Both halves' 13-character names select the Intellibed command table."""
    profile = resolve_motosleep_profile(name)

    assert profile.profile_id == "intellibed_sleep_genius"
    assert profile.transport is MotoSleepTransport.MOTO_BINARY
    assert profile.write_with_response is True


@pytest.mark.parametrize("name", ["HHC14951BEDM", "HHC1495XBEDM9", "HHC14951BEDX9"])
def test_near_intellibed_names_keep_previous_routing(name: str) -> None:
    """Only the exact Intellibed pattern changes routing."""
    assert resolve_motosleep_profile(name).profile_id != "intellibed_sleep_genius"


@pytest.mark.parametrize(
    ("command", "data", "expected"),
    [
        (0x0002, 0, "242300020000024 10d"),
        (0x0001, 0, "242300010000014 10d"),
        (0x0008, 0, "242300080000084 10d"),
        (0x0004, 0, "242300040000044 10d"),
        (0x5555, 0, "242355550000004 10d"),
        (0x8009, 0, "242380090000894 10d"),
        (0x800A, 0, "2423800a00008a4 10d"),
        (0x800B, 0, "2423800b00008b4 10d"),
        (0x800C, 0, "2423800c00008c4 10d"),
        (0x80F1, 0, "242380f10000714 10d"),
        (0x80F2, 0, "242380f20000724 10d"),
        (0x8279, 40, "242382790028d34 10d"),
        (0x8279, 80, "242382790050ab4 10d"),
        (0x8279, 120, "24238279007883 410d"),
        (0x827A, 0, "2423827a0000f84 10d"),
        (0x8100, 0, "242381000000814 10d"),
        (0x80E9, 82, "242380e900523b4 10d"),
        (0x80EA, 119, "242380ea00771d4 10d"),
    ],
)
def test_intellibed_frames_match_app_vectors(command: int, data: int, expected: str) -> None:
    """The shared binary builder reproduces the app's literal frames."""
    assert build_moto_binary_frame(command, data) == bytes.fromhex(expected.replace(" ", ""))


def test_intellibed_capabilities() -> None:
    """Intellibed exposes its app's controls and no MotoSleep extras."""
    controller, _ = _controller(_INTELLIBED_NAME)

    assert set(controller.profile.motors) == {"back", "legs"}
    assert controller.supports_preset_flat
    assert controller.supports_preset_zero_g
    assert controller.supports_preset_anti_snore
    assert controller.supports_preset_tv
    assert controller.supports_preset_lounge
    assert not controller.supports_memory_presets
    assert not controller.supports_memory_programming
    assert not controller.supports_preset_swing
    assert controller.supports_lights
    assert controller.supports_discrete_light_control
    assert controller.supports_explicit_light_on_control
    assert controller.supports_light_color_control
    assert not controller.supports_light_toggle_control
    # Status frames report the light channels, so its state is real.
    assert not controller.light_state_is_assumed
    assert controller.supports_position_feedback
    assert controller.reports_percentage_position
    assert controller.supports_massage
    assert controller.supports_massage_off_control
    assert controller.supports_massage_intensity_control
    assert controller.massage_intensity_zones == ["head", "foot"]
    assert controller.massage_intensity_max == 3
    assert controller.supports_stop_all


async def test_intellibed_head_move_repeats_then_releases() -> None:
    """Held head-up refreshes 0x0002 and ends with five 0x0000 releases."""
    controller, client = _controller(_INTELLIBED_NAME)

    await controller.move_head_up()

    calls = client.write_gatt_char.await_args_list
    assert calls[0].args[1] == build_moto_binary_frame(0x0002)
    assert [c.args[1] for c in calls[1:]] == [build_moto_binary_frame(0x0000)] * 5
    # Refresh frames do not wait for the acknowledgement; the release does.
    assert calls[0].kwargs["response"] is False
    assert all(c.kwargs["response"] is True for c in calls[1:])


async def test_intellibed_feet_down_uses_standard_leg_code() -> None:
    controller, client = _controller(_INTELLIBED_NAME)

    await controller.move_feet_down()

    assert client.write_gatt_char.await_args_list[0].args[1] == build_moto_binary_frame(0x0004)


async def test_intellibed_presets_are_held_then_released() -> None:
    """A single preset frame moves nothing; presets send 20 frames first."""
    controller, client = _controller(_INTELLIBED_NAME)

    with patch.object(controller, "_write_gatt_with_retry", AsyncMock()) as write:
        await controller.preset_zero_g()

    first, release = write.await_args_list
    assert first.args[1] == build_moto_binary_frame(0x800B)
    assert first.kwargs["repeat_count"] == 20
    assert first.kwargs["repeat_delay_ms"] == 100
    assert release.args[1] == build_moto_binary_frame(0x0000)
    assert release.kwargs["repeat_count"] == 5


async def test_intellibed_flat_and_lounge_use_app_codes() -> None:
    controller, _ = _controller(_INTELLIBED_NAME)

    with patch.object(controller, "_write_gatt_with_retry", AsyncMock()) as write:
        await controller.preset_flat()
        await controller.preset_lounge()

    frames = [c.args[1] for c in write.await_args_list]
    assert frames == [
        build_moto_binary_frame(0x5555),
        build_moto_binary_frame(0x0000),
        build_moto_binary_frame(0x800C),
        build_moto_binary_frame(0x0000),
    ]


async def test_intellibed_light_on_off_are_explicit_single_frames() -> None:
    """Hardware-confirmed on/off frames, never a toggle."""
    controller, client = _controller(_INTELLIBED_NAME)

    await controller.lights_on()
    await controller.lights_off()

    assert [c.args[1] for c in client.write_gatt_char.await_args_list] == [
        bytes.fromhex("242380f1000071410d"),
        bytes.fromhex("242380f2000072410d"),
    ]


async def test_intellibed_white_sends_no_channel_frames() -> None:
    """The app's white is the on frame alone; channels at 120 are never sent."""
    controller, client = _controller(_INTELLIBED_NAME)

    with patch("custom_components.adjustable_bed.beds.motosleep.asyncio.sleep"):
        await controller.set_light_color((255, 255, 255))

    client.write_gatt_char.assert_not_awaited()


async def test_intellibed_colour_writes_three_channel_frames() -> None:
    """The app's blue (2, 82, 119) in red, green, blue channel order."""
    controller, client = _controller(_INTELLIBED_NAME)
    rgb = tuple(round(v * 255 / 120) for v in (2, 82, 119))

    with patch("custom_components.adjustable_bed.beds.motosleep.asyncio.sleep"):
        await controller.set_light_color(rgb)

    assert [c.args[1] for c in client.write_gatt_char.await_args_list] == [
        bytes.fromhex("242380e800026a410d"),
        bytes.fromhex("242380e900523b410d"),
        bytes.fromhex("242380ea00771d410d"),
    ]


async def test_intellibed_black_turns_the_light_off() -> None:
    controller, client = _controller(_INTELLIBED_NAME)

    await controller.set_light_color((0, 0, 0))

    client.write_gatt_char.assert_awaited_once_with(
        MOTOSLEEP_CHAR_UUID, bytes.fromhex("242380f2000072410d"), response=True
    )


async def test_intellibed_massage_levels_and_stop_all() -> None:
    controller, client = _controller(_INTELLIBED_NAME)

    await controller.set_massage_intensity("head", 1)
    await controller.set_massage_intensity("foot", 3)
    assert controller.get_massage_state() == {"head_intensity": 1, "foot_intensity": 3}
    await controller.massage_off()

    assert [c.args[1] for c in client.write_gatt_char.await_args_list] == [
        bytes.fromhex("242382790028d3410d"),
        bytes.fromhex("2423827a007880410d"),
        bytes.fromhex("2423810000008141 0d".replace(" ", "")),
    ]
    assert controller.get_massage_state() == {"head_intensity": 0, "foot_intensity": 0}
    with pytest.raises(ValueError):
        await controller.set_massage_intensity("head", 4)


async def test_intellibed_massage_steps_use_app_codes() -> None:
    controller, client = _controller(_INTELLIBED_NAME)

    await controller.massage_head_up()
    await controller.massage_head_down()
    await controller.massage_foot_up()
    await controller.massage_foot_down()

    assert [c.args[1] for c in client.write_gatt_char.await_args_list] == [
        build_moto_binary_frame(command) for command in (0x8102, 0x8104, 0x8103, 0x8105)
    ]


# ----- Intellibed status frames ------------------------------------------------


def _status_line(**fields: int) -> bytes:
    """Build one bed status line: "$" + 36 five-digit fields, CR-terminated."""
    values = [0] * 36
    values[33] = 14951
    for key, value in fields.items():
        values[int(key.removeprefix("f"))] = value
    return ("$" + " ".join(f"{v:05d}" for v in values) + "\r").encode()


def _status_controller() -> tuple[MotoSleepController, MagicMock, list[tuple[str, float]]]:
    controller, client = _controller(_INTELLIBED_NAME)
    seen: list[tuple[str, float]] = []
    controller._notify_callback = lambda axis, value: seen.append((axis, value))
    return controller, client, seen


def test_parse_intellibed_status_rejects_text_lines() -> None:
    assert parse_intellibed_status(_status_line(f2=87).decode())[2] == 87
    for line in ("STOP!", "RESET_RF!", "NO_HL!", "$00001 00002", ""):
        assert parse_intellibed_status(line) is None


def test_status_frames_split_across_notifications_report_percent_positions() -> None:
    """Frames arrive in 16-byte chunks mixed with the bed's text lines."""
    controller, _, seen = _status_controller()
    stream = b"\r\nSTOP!\r\n" + _status_line(f2=850, f3=225) + b"RESET_RF!\r\n"

    for start in range(0, len(stream), 16):
        controller.feed_status_bytes(stream[start : start + 16])

    full_back = INTELLIBED_FULL_TRAVEL["back"]
    full_legs = INTELLIBED_FULL_TRAVEL["legs"]
    assert seen == [
        ("back", round(850 / full_back * 100, 1)),
        ("legs", round(225 / full_legs * 100, 1)),
    ]
    assert controller.protocol_diagnostics["status_counts"] == {"back": 850, "legs": 225}


def test_repeated_identical_frames_do_not_repeat_callbacks() -> None:
    controller, _, seen = _status_controller()

    for _ in range(5):
        controller.feed_status_bytes(_status_line(f2=100, f3=0))

    assert len(seen) == 2


def test_counts_beyond_provisional_travel_recalibrate_to_100_percent() -> None:
    controller, _, seen = _status_controller()
    high = INTELLIBED_FULL_TRAVEL["back"] + 300

    controller.feed_status_bytes(_status_line(f2=high))
    controller.feed_status_bytes(_status_line(f2=high // 2))

    back = [value for axis, value in seen if axis == "back"]
    assert back == [100.0, round((high // 2) / high * 100, 1)]
    assert controller.protocol_diagnostics["full_travel"]["back"] == high


def test_status_frames_publish_light_and_massage_state_on_change() -> None:
    controller, _, _ = _status_controller()
    coordinator = controller._coordinator

    controller.feed_status_bytes(_status_line(f6=40, f7=120, f8=120, f9=120, f10=120))
    controller.feed_status_bytes(_status_line(f6=40, f7=120, f8=120, f9=120, f10=120))
    controller.feed_status_bytes(_status_line())

    updates = [c.args[0] for c in coordinator.handle_controller_state_updates.call_args_list]
    assert updates == [
        {
            "intellibed_massage": (1, 3),
            "under_bed_lights_on": True,
            "under_bed_lights_rgb": (255, 255, 255),
        },
        {"intellibed_massage": (0, 0), "under_bed_lights_on": False},
    ]
    assert controller.get_massage_state() == {"head_intensity": 0, "foot_intensity": 0}


async def test_start_notify_subscribes_to_ffe1_once() -> None:
    controller, client = _controller(_INTELLIBED_NAME)
    client.start_notify = AsyncMock()

    await controller.start_notify(lambda axis, value: None)
    await controller.start_notify(lambda axis, value: None)

    client.start_notify.assert_awaited_once()
    assert client.start_notify.await_args.args[0] == MOTOSLEEP_CHAR_UUID


async def test_other_motosleep_profiles_do_not_subscribe() -> None:
    controller, client = _controller(_hhc_name("G", c22="M"))
    client.start_notify = AsyncMock()

    await controller.start_notify(lambda axis, value: None)

    client.start_notify.assert_not_awaited()
    assert not controller.supports_position_feedback


def test_intellibed_position_specs_are_percent_back_and_legs() -> None:
    controller, _ = _controller(_INTELLIBED_NAME)

    specs = controller.position_number_specs

    assert [
        (spec.position_key, spec.native_max_value, spec.native_unit_of_measurement)
        for spec in specs
    ] == [
        ("back", 100, "%"),
        ("legs", 100, "%"),
    ]


def test_intellibed_bed_type_has_position_feedback() -> None:
    assert bed_type_has_position_feedback(BED_TYPE_INTELLIBED, None)
    assert not bed_type_has_position_feedback(BED_TYPE_MOTOSLEEP, None)


@pytest.mark.parametrize("name", ["HHC14951BEDM9", "HHC23235BEDM9"])
def test_intellibed_names_detect_as_their_own_bed_type(name: str) -> None:
    info = MagicMock()
    info.name = name
    info.device.name = name
    info.address = "EC:C5:7F:3A:2B:A1"
    info.service_uuids = ["0000180a-0000-1000-8000-00805f9b34fb"]
    info.manufacturer_data = {}
    info.rssi = -60

    assert detect_bed_type(info) == BED_TYPE_INTELLIBED


def test_other_hhc_names_still_detect_as_motosleep() -> None:
    info = MagicMock()
    info.name = "HHC0069815CDEF"
    info.device.name = "HHC0069815CDEF"
    info.address = "AA:BB:CC:DD:EE:FF"
    info.service_uuids = []
    info.manufacturer_data = {}
    info.rssi = -60

    assert detect_bed_type(info) == BED_TYPE_MOTOSLEEP


async def test_factory_builds_intellibed_controller(hass) -> None:
    coordinator = MagicMock(hass=hass)
    controller = await create_controller(
        coordinator,
        BED_TYPE_INTELLIBED,
        protocol_variant=None,
        client=None,
        device_name=None,
    )

    assert isinstance(controller, MotoSleepController)
    assert controller.profile.profile_id == "intellibed_sleep_genius"


# ----- Intellibed held controls ----------------------------------------------


def test_intellibed_motors_declare_continuous_holds() -> None:
    controller, _ = _controller(_INTELLIBED_NAME)

    holds = {spec.key: spec.hold for spec in controller.motor_control_specs}

    assert controller.held_control_options == ("head_up", "head_down", "feet_up", "feet_down")
    assert (holds["back"].up_control, holds["back"].down_control) == ("head_up", "head_down")
    assert (holds["legs"].up_control, holds["legs"].down_control) == ("feet_up", "feet_down")


def test_other_motosleep_profiles_keep_finite_pulses() -> None:
    controller, _ = _controller(_hhc_name("G", c22="M"))

    assert controller.held_control_options == ()
    assert all(spec.hold is None for spec in controller.motor_control_specs)


async def test_intellibed_hold_refreshes_until_duration_then_releases() -> None:
    controller, _ = _controller(_INTELLIBED_NAME)

    with patch.object(controller, "_write_gatt_with_retry", AsyncMock()) as write:
        await controller.hold_control("feet_up", 1000)

    move, release = write.await_args_list
    assert move.args[1] == build_moto_binary_frame(0x0008)
    assert move.kwargs["repeat_count"] == 10
    assert move.kwargs["repeat_delay_ms"] == 100
    assert release.args[1] == build_moto_binary_frame(0x0000)
    assert release.kwargs["repeat_count"] == 5


async def test_intellibed_hold_stops_at_deadline_and_still_releases() -> None:
    """A slow link cannot stretch a hold past its duration."""
    controller, _ = _controller(_INTELLIBED_NAME)
    frames: list[bytes] = []

    async def slow_write(_uuid, frame, **kwargs):
        frames.append(frame)
        if frame == build_moto_binary_frame(0x0002):
            await asyncio.sleep(10)

    with patch.object(controller, "_write_gatt_with_retry", side_effect=slow_write):
        await controller.hold_control("head_up", 50)

    assert frames == [build_moto_binary_frame(0x0002), build_moto_binary_frame(0x0000)]


@pytest.mark.parametrize(
    ("control", "duration"), [("lights", 1000), ("head_up", 0), ("head_up", 60001)]
)
def test_intellibed_hold_rejects_bad_requests(control: str, duration: int) -> None:
    controller, _ = _controller(_INTELLIBED_NAME)

    with pytest.raises(ValueError):
        controller.validate_hold_control(control, duration)


async def test_intellibed_hold_refresh_is_wall_clock_paced_without_response() -> None:
    """A slow GATT acknowledgement must not stretch the refresh interval."""
    controller, _ = _controller(_INTELLIBED_NAME)

    with patch.object(controller, "_write_gatt_with_retry", AsyncMock()) as write:
        await controller.hold_control("head_up", 4000)

    move = write.await_args_list[0]
    assert move.args[1] == build_moto_binary_frame(0x0002)
    assert move.kwargs["repeat_count"] == 40
    assert move.kwargs["repeat_delay_ms"] == 100
    assert move.kwargs["response"] is False
    assert move.kwargs["wall_clock_pacing"] is True
    assert callable(move.kwargs["on_write"])


async def test_intellibed_refresh_logs_slow_gaps(caplog: pytest.LogCaptureFixture) -> None:
    controller, _ = _controller(_INTELLIBED_NAME)
    times = iter([0.0, 0.0, 0.1, 0.9])

    async def fake_write(*_args, on_write, repeat_count, **_kwargs):
        for _ in range(repeat_count):
            on_write()

    loop = asyncio.get_running_loop()
    with (
        patch.object(controller, "_write_gatt_with_retry", side_effect=fake_write),
        patch.object(loop, "time", side_effect=lambda: next(times)),
        caplog.at_level("DEBUG", logger="custom_components.adjustable_bed.beds.motosleep"),
    ):
        await controller._send_refresh(0x0002, repeat_count=3, repeat_delay_ms=100)

    assert "Refresh frame 3 of 3 sent 800 ms after the previous one" in caplog.text


def test_intellibed_stop_lines_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    controller, _ = _controller(_INTELLIBED_NAME)

    with caplog.at_level("DEBUG", logger="custom_components.adjustable_bed.beds.motosleep"):
        controller.feed_status_bytes(b"\r\nSTOP!\r\n")

    assert "reported STOP!" in caplog.text


async def test_non_intellibed_moves_keep_acknowledged_writes() -> None:
    controller, client = _controller(_hhc_name("G", c22="M"))

    await controller.move_head_up()

    assert all(
        c.kwargs["response"] is controller.profile.write_with_response
        for c in client.write_gatt_char.await_args_list
    )
