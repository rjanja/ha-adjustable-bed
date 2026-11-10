"""MotoSleep, Power Bob, and MOTO adjustable-bed controller."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from bleak.exc import BleakError

from ..const import MOTOSLEEP_CHAR_UUID
from .base import (
    POSITION_UNIT_PERCENT,
    BedController,
    MotorCommandCallable,
    MotorControlSpec,
    PositionNumberSpec,
    build_position_number_spec,
)
from .motosleep_profiles import (
    Command,
    MotoSleepProfile,
    MotoSleepTransport,
    resolve_motosleep_profile,
)

if TYPE_CHECKING:
    from bleak.backends.characteristic import BleakGATTCharacteristic

    from ..coordinator import AdjustableBedCoordinator

_LOGGER = logging.getLogger(__name__)

_FFE1 = "0000ffe1-0000-1000-8000-00805f9b34fb"
_FFF1 = "0000fff1-0000-1000-8000-00805f9b34fb"

# Intellibed status frames: "$" + 36 space-separated five-digit decimal fields,
# ending CR, about five a second on FFE1.  Field indexes (zero-based):
_STATUS_FIELD_COUNT = 36
_STATUS_HEAD = 2  # head motor position count
_STATUS_FEET = 3  # feet motor position count
_STATUS_HEAD_MASSAGE = 6  # 0 / 40 / 80 / 120
_STATUS_FOOT_MASSAGE = 7
_STATUS_LIGHT_RGB = (8, 9, 10)  # 0..120 per channel
_STATUS_CHANNEL_MAX = 120
# Provisional full-travel counts.  Observed on one base: head up to 1655 during
# zero-G, feet up to 232.  The scale also grows to the largest count seen, so a
# full raise self-calibrates; percentages are clamped to 100.
INTELLIBED_FULL_TRAVEL: dict[str, int] = {"back": 1700, "legs": 450}
_STATUS_BUFFER_LIMIT = 512


def parse_intellibed_status(line: str) -> list[int] | None:
    """Return the 36 fields of one Intellibed status line, or None."""
    line = line.strip()
    if not line.startswith("$"):
        return None
    tokens = line[1:].split()
    if len(tokens) != _STATUS_FIELD_COUNT or not all(t.isdigit() for t in tokens):
        return None
    return [int(token) for token in tokens]


class MotoSleepCommands:
    """Stable ASCII command constants retained for API compatibility."""

    PRESET_HOME = ord("O")
    PRESET_MEMORY_1 = ord("U")
    PRESET_MEMORY_2 = ord("V")
    PRESET_ANTI_SNORE = ord("R")
    PRESET_TV = ord("S")
    PRESET_ZERO_G = ord("T")
    PROGRAM_MEMORY_1 = ord("Z")
    PROGRAM_MEMORY_2 = ord("a")
    MOTOR_HEAD_UP = ord("K")
    MOTOR_HEAD_DOWN = ord("L")
    MOTOR_FEET_UP = ord("M")
    MOTOR_FEET_DOWN = ord("N")
    MOTOR_NECK_UP = ord("P")
    MOTOR_NECK_DOWN = ord("Q")
    MOTOR_LUMBAR_UP = ord("p")
    MOTOR_LUMBAR_DOWN = ord("q")
    MASSAGE_HEAD_STEP = ord("C")
    MASSAGE_FOOT_STEP = ord("B")
    MASSAGE_STOP = ord("D")
    MASSAGE_HEAD_UP = ord("G")
    MASSAGE_HEAD_DOWN = ord("H")
    MASSAGE_FOOT_UP = ord("E")
    MASSAGE_FOOT_DOWN = ord("F")
    STOP = ord("b")
    LIGHTS_TOGGLE = ord("A")


def build_moto_binary_frame(command: int, data: int = 0) -> bytes:
    """Build the exact nine-byte MOTO frame recovered from the APK."""
    command &= 0xFFFF
    data &= 0xFFFF
    body = bytes((command >> 8, command & 0xFF, data >> 8, data & 0xFF))
    checksum = body[0] ^ body[1] ^ body[2] ^ body[3]
    return b"$#" + body + bytes((checksum, 0x41, 0x0D))


def _xor_decimal_digits(value: str) -> int:
    checksum = 0
    for digit in value:
        checksum ^= int(digit)
    return checksum


def build_power_bob_numeric_frame(selector: str, value: str) -> bytes:
    """Build Power Bob's selector-then-value decimal frame."""
    payload = selector + value
    return f"$#{payload}{_xor_decimal_digits(payload):05d}R\r".encode()


def build_motosleep_numeric_frame(value: str, selector: str) -> bytes:
    """Build MotoSleep's value-then-selector decimal frame."""
    payload = value + selector
    return f"$#{payload}{_xor_decimal_digits(payload):05d}R\r".encode()


def build_motosleep_sync_frame(opcode: str) -> bytes:
    """Build the current MotoSleep MainPanel sync/bind frame."""
    payload = opcode + "00000"
    return f"$#{payload}{_xor_decimal_digits(payload):05d}R\r".encode()


class MotoSleepController(BedController):
    """Profile-driven controller matching both shipped HHC applications."""

    def __init__(
        self,
        coordinator: AdjustableBedCoordinator,
        device_name: str | None = None,
        profile: MotoSleepProfile | None = None,
    ) -> None:
        super().__init__(coordinator)
        self._device_name = device_name or getattr(coordinator, "ble_device_name", None)
        self.profile: MotoSleepProfile = profile or resolve_motosleep_profile(self._device_name)
        self._light_rgb = (255, 255, 255)
        self._massage_levels: dict[str, int] = {}
        self._status_buffer = ""
        self._status_notifying = False
        self._status_counts: dict[str, int] = {}
        self._status_max_seen: dict[str, int] = {}
        self._status_positions: dict[str, float] = {}
        self._status_published: dict[str, Any] = {}
        _LOGGER.info(
            "MotoSleep local name %r resolved to profile %s",
            self._device_name,
            self.profile.profile_id,
        )

    @property
    def control_characteristic_uuid(self) -> str:
        """Prefer the app's FFE1 role, with its FFF1 fallback."""
        client = self.client
        if client is not None and client.services:
            for candidate in (_FFE1, _FFF1):
                for service in client.services:
                    for characteristic in service.characteristics:
                        if str(characteristic.uuid).lower() == candidate:
                            return str(characteristic.uuid)
        return MOTOSLEEP_CHAR_UUID

    @property
    def supports_position_feedback(self) -> bool:
        return self.profile.status_frames

    @property
    def reports_percentage_position(self) -> bool:
        return self.profile.status_frames

    @property
    def position_number_specs(self) -> tuple[PositionNumberSpec, ...]:
        if not self.profile.status_frames:
            return super().position_number_specs
        return (
            build_position_number_spec("back", max_value=100, unit=POSITION_UNIT_PERCENT),
            build_position_number_spec("legs", max_value=100, unit=POSITION_UNIT_PERCENT),
        )

    @property
    def protocol_diagnostics(self) -> dict[str, Any]:
        if not self.profile.status_frames:
            return {}
        return {
            "profile": self.profile.profile_id,
            "status_counts": dict(self._status_counts),
            "status_max_seen": dict(self._status_max_seen),
            "full_travel": {axis: self._full_travel(axis) for axis in INTELLIBED_FULL_TRAVEL},
        }

    @property
    def auto_stops_on_idle(self) -> bool:
        return self.profile.stop is None

    @property
    def supports_stop_all(self) -> bool:
        return self.profile.stop is not None

    @property
    def supports_preset_flat(self) -> bool:
        return "flat" in self.profile.presets

    @property
    def supports_preset_zero_g(self) -> bool:
        return "zero_g" in self.profile.presets

    @property
    def supports_preset_anti_snore(self) -> bool:
        return "anti_snore" in self.profile.presets

    @property
    def supports_preset_tv(self) -> bool:
        return "tv" in self.profile.presets

    @property
    def supports_preset_lounge(self) -> bool:
        return "lounge" in self.profile.presets

    @property
    def supports_preset_swing(self) -> bool:
        return self.profile.swing is not None

    @property
    def supports_memory_presets(self) -> bool:
        return bool(self.profile.memory_recall)

    @property
    def memory_slot_count(self) -> int:
        return len(self.profile.memory_recall)

    @property
    def supports_memory_programming(self) -> bool:
        return bool(self.profile.memory_save)

    @property
    def _has_explicit_light(self) -> bool:
        return self.profile.light_on is not None and self.profile.light_off is not None

    @property
    def supports_lights(self) -> bool:
        return (
            self.profile.light_toggle is not None
            or self.profile.rgb_light
            or self._has_explicit_light
        )

    @property
    def supports_light_color_control(self) -> bool:
        return self.profile.rgb_light or self.profile.binary_rgb_channels is not None

    @property
    def supports_light_toggle_control(self) -> bool:
        return self.profile.light_toggle is not None

    @property
    def supported_color_mode(self) -> str | None:
        return "rgb" if self.supports_light_color_control else None

    @property
    def default_light_rgb_color(self) -> tuple[int, int, int] | None:
        return (255, 255, 255) if self.supports_light_color_control else None

    @property
    def supports_discrete_light_control(self) -> bool:
        # RGB channels are absolute: a non-zero colour turns the LEDs on and
        # three zero channels turn them off.  Treat that as discrete power so
        # Home Assistant does not try the unrelated $A toggle on RGB profiles.
        return self.profile.rgb_light or self._has_explicit_light

    @property
    def supports_explicit_light_on_control(self) -> bool:
        # Numeric RGB: set_light_color() itself is the idempotent on operation.
        # Returning False avoids sending the three RGB channels twice.  Binary
        # explicit-light profiles send their on frame before any colour.
        return self._has_explicit_light

    @property
    def light_state_is_assumed(self) -> bool:
        # Status frames report the light channels, so the state is real.
        return self._has_explicit_light and not self.profile.status_frames

    @property
    def supports_massage_intensity_control(self) -> bool:
        return bool(self.profile.massage_levels)

    @property
    def massage_intensity_zones(self) -> list[str]:
        return list(self.profile.massage_levels)

    @property
    def massage_intensity_max(self) -> int:
        return self.profile.massage_level_max or super().massage_intensity_max

    @property
    def supports_massage(self) -> bool:
        return bool(self.profile.massage) or bool(self.profile.massage_levels)

    @property
    def supports_synchro(self) -> bool:
        return self.profile.synchro is not None

    @property
    def supports_auxiliary_action(self) -> bool:
        return self.profile.auxiliary_action is not None

    @property
    def auto_enable_massage(self) -> bool:
        """Expose app-proven massage controls without a legacy manual toggle."""
        return self.supports_massage

    @property
    def supports_massage_off_control(self) -> bool:
        return "off" in self.profile.massage

    @property
    def supports_head_massage_toggle_control(self) -> bool:
        return "head_toggle" in self.profile.massage

    @property
    def supports_head_massage_intensity_step_control(self) -> bool:
        return {"head_up", "head_down"} <= self.profile.massage.keys()

    @property
    def supports_foot_massage_toggle_control(self) -> bool:
        return "foot_toggle" in self.profile.massage

    @property
    def supports_foot_massage_intensity_step_control(self) -> bool:
        return {"foot_up", "foot_down"} <= self.profile.massage.keys()

    @property
    def has_lumbar_support(self) -> bool:
        return "lumbar" in self.profile.motors

    @property
    def has_neck_support(self) -> bool:
        return "neck" in self.profile.motors

    @property
    def has_tilt_support(self) -> bool:
        return "tilt" in self.profile.motors

    @property
    def stale_motor_entity_keys(self) -> frozenset[str]:
        return frozenset(
            {
                "back",
                "legs",
                "head",
                "feet",
                "lumbar",
                "neck",
                "tilt",
                "auxiliary",
                "auxiliary_1",
                "auxiliary_2",
            }
        )

    @property
    def motor_control_specs(self) -> tuple[MotorControlSpec, ...]:
        """Expose exactly the actuators rendered by the resolved app page."""

        def _open(key: str) -> MotorCommandCallable:
            async def command(controller: BedController) -> None:
                await cast(MotoSleepController, controller)._move_motor(key, True)

            return command

        def _close(key: str) -> MotorCommandCallable:
            async def command(controller: BedController) -> None:
                await cast(MotoSleepController, controller)._move_motor(key, False)

            return command

        async def _stop(controller: BedController) -> None:
            await cast(MotoSleepController, controller)._send_stop()

        return tuple(
            MotorControlSpec(
                key=key,
                translation_key=key,
                open_fn=_open(key),
                close_fn=_close(key),
                stop_fn=_stop,
                max_angle=45 if key in {"legs", "feet", "tilt"} else 68,
            )
            for key in self.profile.motors
        )

    def _build_command(self, char_code: int) -> bytes:
        """Build an ASCII action according to the resolved HHC transport."""
        return self._encode_action(chr(char_code))

    def _encode_action(self, action: Command, data: int = 0) -> bytes:
        if self.profile.transport is MotoSleepTransport.MOTO_BINARY:
            if not isinstance(action, int):
                raise TypeError("MOTO binary actions must be integers")
            return build_moto_binary_frame(action, data)

        if not isinstance(action, str) or len(action) != 1:
            raise TypeError("HHC actions must be one ASCII character")
        raw = f"${action}".encode()
        if self.profile.transport is MotoSleepTransport.POWER_BOB_ASCII:
            return raw
        if self.profile.raw_hhc:
            return raw
        # The APK's wrapped path also drops the special $z query.  The
        # integration does not send that initialization-only action.
        return b"$#" + raw + b"R\r"

    async def write_command(
        self,
        command: bytes,
        repeat_count: int = 1,
        repeat_delay_ms: int = 100,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        """Write with the response mode selected by the OEM app."""
        await self._write_gatt_with_retry(
            self.control_characteristic_uuid,
            command,
            repeat_count=repeat_count,
            repeat_delay_ms=repeat_delay_ms,
            cancel_event=cancel_event,
            response=self.profile.write_with_response,
        )

    # ----- Intellibed status frames -------------------------------------

    def _full_travel(self, axis: str) -> int:
        return max(INTELLIBED_FULL_TRAVEL[axis], self._status_max_seen.get(axis, 0))

    async def start_notify(self, callback: Callable[[str, float], None] | None = None) -> None:
        """Subscribe to FFE1 status frames on profiles that stream them."""
        self._notify_callback = callback
        if not self.profile.status_frames or self._status_notifying:
            return
        client = self.client
        if client is None or not client.is_connected:
            _LOGGER.debug("Cannot start Intellibed status notifications: not connected")
            return
        try:
            async with self._ble_lock:
                await client.start_notify(
                    self.control_characteristic_uuid, self._handle_status_notification
                )
        except BleakError as err:
            _LOGGER.debug("Could not start Intellibed status notifications: %s", err)
            return
        self._status_notifying = True
        self._status_buffer = ""

    async def stop_notify(self) -> None:
        client = self.client
        try:
            if self._status_notifying and client is not None and client.is_connected:
                async with self._ble_lock:
                    await client.stop_notify(self.control_characteristic_uuid)
        except BleakError:
            _LOGGER.debug("Failed to stop Intellibed status notifications", exc_info=True)
        finally:
            self._status_notifying = False
            self._status_buffer = ""
            self._notify_callback = None

    def on_disconnect(self) -> None:
        self._status_notifying = False
        self._status_buffer = ""
        self._status_published = {}

    def _handle_status_notification(
        self, _characteristic: BleakGATTCharacteristic | Any, data: bytearray | bytes
    ) -> None:
        payload = bytes(data)
        self.forward_raw_notification(self.control_characteristic_uuid, payload)
        self.feed_status_bytes(payload)

    def feed_status_bytes(self, payload: bytes) -> None:
        """Reassemble CR/LF-terminated lines and apply each status frame."""
        self._status_buffer += payload.decode("ascii", errors="ignore")
        *lines, self._status_buffer = self._status_buffer.replace("\n", "\r").split("\r")
        if len(self._status_buffer) > _STATUS_BUFFER_LIMIT:
            self._status_buffer = ""
        for line in lines:
            fields = parse_intellibed_status(line)
            if fields is not None:
                self._apply_status(fields)

    def _apply_status(self, fields: list[int]) -> None:
        for axis, index in (("back", _STATUS_HEAD), ("legs", _STATUS_FEET)):
            count = fields[index]
            self._status_counts[axis] = count
            if count > self._status_max_seen.get(axis, 0):
                self._status_max_seen[axis] = count
            percent = round(min(100.0, count / self._full_travel(axis) * 100), 1)
            if self._status_positions.get(axis) != percent:
                self._status_positions[axis] = percent
                if self._notify_callback is not None:
                    self._notify_callback(axis, percent)

        step = self.profile.massage_level_step or 40
        levels = {
            "head": min(self.profile.massage_level_max, fields[_STATUS_HEAD_MASSAGE] // step),
            "foot": min(self.profile.massage_level_max, fields[_STATUS_FOOT_MASSAGE] // step),
        }
        rgb_raw = tuple(fields[index] for index in _STATUS_LIGHT_RGB)
        light_on = any(rgb_raw)
        updates: dict[str, Any] = {}
        if levels != self._massage_levels:
            self._massage_levels = levels
            updates["intellibed_massage"] = (levels["head"], levels["foot"])
        if self._status_published.get("under_bed_lights_on") != light_on:
            updates["under_bed_lights_on"] = light_on
        if light_on:
            rgb = tuple(
                round(min(channel, _STATUS_CHANNEL_MAX) / _STATUS_CHANNEL_MAX * 255)
                for channel in rgb_raw
            )
            if self._status_published.get("under_bed_lights_rgb") != rgb:
                updates["under_bed_lights_rgb"] = rgb
                self._light_rgb = cast(tuple[int, int, int], rgb)
        if updates:
            self._status_published.update(updates)
            self.forward_controller_state_updates(updates)

    def get_status_positions(self) -> dict[str, float]:
        return dict(self._status_positions)

    async def read_positions(self, motor_count: int = 2) -> None:  # noqa: ARG002
        """Positions stream continuously; re-announce the latest values."""
        if self._notify_callback is None:
            return
        for axis, percent in self._status_positions.items():
            self._notify_callback(axis, percent)

    async def _send_action(
        self,
        action: Command,
        *,
        data: int = 0,
        repeat_count: int = 1,
        repeat_delay_ms: int = 100,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        await self.write_command(
            self._encode_action(action, data),
            repeat_count=repeat_count,
            repeat_delay_ms=repeat_delay_ms,
            cancel_event=cancel_event,
        )

    async def _send_stop(self) -> None:
        stop = self.profile.stop
        if stop is None:
            return
        action, repeat_count, repeat_delay_ms = stop
        data = 0
        if isinstance(action, tuple):
            action, data = action
        await self._send_action(
            action,
            data=data,
            repeat_count=repeat_count,
            repeat_delay_ms=repeat_delay_ms,
            cancel_event=asyncio.Event(),
        )

    async def _move_motor(self, key: str, upward: bool) -> None:
        pair = self.profile.motors.get(key)
        if pair is None:
            raise NotImplementedError(f"{key} motor is not present on {self.profile.profile_id}")
        try:
            if self.profile.transport is MotoSleepTransport.POWER_BOB_ASCII:
                # Power Bob's repeating ImageButton has no immediate write;
                # its first movement pulse fires after one 100 ms interval.
                await asyncio.sleep(0.1)
            await self._send_action(
                pair[0 if upward else 1],
                repeat_count=self._coordinator.motor_pulse_count,
                repeat_delay_ms=100,
            )
        finally:
            await self._send_stop()

    async def move_head_up(self) -> None:
        await self._move_motor("head" if "head" in self.profile.motors else "back", True)

    async def move_head_down(self) -> None:
        await self._move_motor("head" if "head" in self.profile.motors else "back", False)

    async def move_head_stop(self) -> None:
        await self._send_stop()

    async def move_back_up(self) -> None:
        await self._move_motor("back", True)

    async def move_back_down(self) -> None:
        await self._move_motor("back", False)

    async def move_back_stop(self) -> None:
        await self._send_stop()

    async def move_legs_up(self) -> None:
        await self._move_motor("legs", True)

    async def move_legs_down(self) -> None:
        await self._move_motor("legs", False)

    async def move_legs_stop(self) -> None:
        await self._send_stop()

    async def move_feet_up(self) -> None:
        await self._move_motor("feet" if "feet" in self.profile.motors else "legs", True)

    async def move_feet_down(self) -> None:
        await self._move_motor("feet" if "feet" in self.profile.motors else "legs", False)

    async def move_feet_stop(self) -> None:
        await self._send_stop()

    async def move_neck_up(self) -> None:
        await self._move_motor("neck", True)

    async def move_neck_down(self) -> None:
        await self._move_motor("neck", False)

    async def move_neck_stop(self) -> None:
        await self._send_stop()

    async def move_lumbar_up(self) -> None:
        await self._move_motor("lumbar", True)

    async def move_lumbar_down(self) -> None:
        await self._move_motor("lumbar", False)

    async def move_lumbar_stop(self) -> None:
        await self._send_stop()

    async def move_tilt_up(self) -> None:
        await self._move_motor("tilt", True)

    async def move_tilt_down(self) -> None:
        await self._move_motor("tilt", False)

    async def move_tilt_stop(self) -> None:
        await self._send_stop()

    async def stop_all(self) -> None:
        await self._send_stop()

    async def _preset(self, key: str) -> None:
        action = self.profile.presets.get(key)
        if action is None:
            raise NotImplementedError(f"{key} preset is not present on {self.profile.profile_id}")
        if self.profile.preset_repeat <= 1:
            await self._send_action(action)
            return
        try:
            await self._send_action(
                action,
                repeat_count=max(self.profile.preset_repeat, self._coordinator.motor_pulse_count),
                repeat_delay_ms=100,
            )
        finally:
            await self._send_stop()

    async def preset_lounge(self) -> None:
        await self._preset("lounge")

    async def preset_flat(self) -> None:
        await self._preset("flat")

    async def preset_zero_g(self) -> None:
        await self._preset("zero_g")

    async def preset_anti_snore(self) -> None:
        await self._preset("anti_snore")

    async def preset_tv(self) -> None:
        await self._preset("tv")

    async def preset_swing(self) -> None:
        if self.profile.swing is None:
            raise NotImplementedError("Swing is not present on this model")
        await self._send_action(self.profile.swing)

    async def set_synchro(self, enabled: bool) -> None:
        if self.profile.synchro is None:
            raise NotImplementedError("Synchronization is not present on this model")
        opcode = self.profile.synchro[0 if enabled else 1]
        await self.write_command(build_motosleep_sync_frame(opcode))

    async def auxiliary_action(self) -> None:
        """Send a profile-specific action whose physical label is not proven."""
        if self.profile.auxiliary_action is None:
            raise NotImplementedError("Auxiliary action is not present on this model")
        await self._send_action(self.profile.auxiliary_action)

    async def preset_memory(self, memory_num: int) -> None:
        if not 1 <= memory_num <= len(self.profile.memory_recall):
            raise ValueError(f"Unsupported memory slot: {memory_num}")
        await self._send_action(self.profile.memory_recall[memory_num - 1])

    async def program_memory(self, memory_num: int) -> None:
        if not 1 <= memory_num <= len(self.profile.memory_save):
            raise ValueError(f"Unsupported memory slot: {memory_num}")
        await self._send_action(self.profile.memory_save[memory_num - 1])

    async def lights_on(self) -> None:
        if self.profile.light_on is not None:
            await self._send_action(self.profile.light_on)
            return
        if self.profile.rgb_light:
            await self.set_light_color(self._light_rgb)
            return
        await self.lights_toggle()

    async def lights_off(self) -> None:
        if self.profile.light_off is not None:
            await self._send_action(self.profile.light_off)
            return
        if self.profile.rgb_light:
            await self.set_light_color((0, 0, 0))
            return
        await self.lights_toggle()

    async def lights_toggle(self) -> None:
        if self.profile.light_toggle is None:
            raise NotImplementedError("Light toggle is not present on this model")
        await self._send_action(self.profile.light_toggle)

    async def set_light_color(self, rgb_color: tuple[int, int, int]) -> None:
        channels = self.profile.binary_rgb_channels
        if channels is not None:
            await self._set_binary_light_color(channels, rgb_color)
            return
        if not self.profile.rgb_light:
            raise NotImplementedError("RGB light is not present on this model")
        scaled = tuple(int(channel / 255 * 120) for channel in rgb_color)
        frames = []
        for selector, channel in zip(("00315", "00316", "00317"), scaled, strict=True):
            value = f"{channel:05d}"
            if self.profile.transport is MotoSleepTransport.POWER_BOB_ASCII:
                frames.append(build_power_bob_numeric_frame(selector, value))
            else:
                frames.append(build_motosleep_numeric_frame(value, selector))
        for index, frame in enumerate(frames):
            await self.write_command(frame)
            if index < 2:
                await asyncio.sleep(0.02)
        if any(rgb_color):
            self._light_rgb = rgb_color

    async def _set_binary_light_color(
        self, channels: tuple[int, int, int], rgb_color: tuple[int, int, int]
    ) -> None:
        """Write the app's three channel frames (0..120) after the on frame.

        The app's white button sends only the light-on frame and no channel
        values, so white is left to lights_on().  A black colour is light off.
        """
        if not any(rgb_color):
            await self.lights_off()
            return
        if tuple(rgb_color) != (255, 255, 255):
            scaled = tuple(round(channel / 255 * 120) for channel in rgb_color)
            for command, value in zip(channels, scaled, strict=True):
                await asyncio.sleep(0.1)
                await self._send_action(command, data=value)
        self._light_rgb = rgb_color

    def get_light_state(self) -> dict[str, object]:
        return {"rgb_color": self._light_rgb}

    async def set_massage_intensity(self, zone: str, level: int) -> None:
        command = self.profile.massage_levels.get(zone)
        if command is None:
            raise NotImplementedError(f"{zone} massage intensity is not present")
        if not 0 <= level <= self.profile.massage_level_max:
            raise ValueError(f"Massage level must be 0-{self.profile.massage_level_max}")
        await self._send_action(command, data=level * self.profile.massage_level_step)
        self._massage_levels[zone] = level

    def get_massage_state(self) -> dict[str, object]:
        return {f"{zone}_intensity": level for zone, level in self._massage_levels.items()}

    async def _massage(self, key: str) -> None:
        action = self.profile.massage.get(key)
        if action is None:
            raise NotImplementedError(f"{key} massage control is not present")
        await self._send_action(action)

    async def massage_off(self) -> None:
        await self._massage("off")
        self._massage_levels = dict.fromkeys(self.profile.massage_levels, 0)

    async def massage_head_toggle(self) -> None:
        await self._massage("head_toggle")

    async def massage_foot_toggle(self) -> None:
        await self._massage("foot_toggle")

    async def massage_head_up(self) -> None:
        await self._massage("head_up")

    async def massage_head_down(self) -> None:
        await self._massage("head_down")

    async def massage_foot_up(self) -> None:
        await self._massage("foot_up")

    async def massage_foot_down(self) -> None:
        await self._massage("foot_down")
