# ocr_pokemon.py
# pip install pywin32 mss numpy opencv-python pytesseract psutil pydirectinput pynput
#
# Files:
# - pokemon.txt  (all pokemon names, one per line)
# - targets.txt  (auto-written by picker)
#
# Hotkeys (global):
# - ESC: Stop
# - P: Pause/Unpause (hard pause, no keys, no clicks)
# - R: Resume farming after you caught the target (clears alert state)
# - O: Open picker and reload targets mid-run

from __future__ import annotations

import time
import ctypes
import re
import threading
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import mss
import cv2
import pytesseract
import psutil
import pydirectinput as pdi

import win32gui
import win32con
import win32process
import win32api
import win32ui
import winsound

from pynput import keyboard

# ----------------------------
# DPI awareness
# ----------------------------
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

# ----------------------------
# pydirectinput settings
# ----------------------------
pdi.FAILSAFE = False
pdi.PAUSE = 0.0

# ----------------------------
# Window standardisation
# ----------------------------
WIN_X, WIN_Y = 50, 50
WIN_W, WIN_H = 1280, 720

# ----------------------------
# Encounter detection (Fight button pixel)
# ----------------------------
FIGHT_ANCHOR_XY = (836, 537)
FIGHT_EXPECTED_RGB = (102, 38, 38)  # #662626
FIGHT_TOL = 25
ENCOUNTER_POLL_S = 0.50

# ----------------------------
# Buttons
# ----------------------------
RUN_CLICK = (830, 622)

# ----------------------------
# OCR ROI
# ----------------------------
POKENAME_ROI = (0.758, 0.222, 0.842, 0.268)

TESS_CONFIG = (
    "--psm 7 "
    "--oem 3 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)
# If tesseract.exe is not on PATH:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------------------
# Matching rules
# ----------------------------
STABLE_FRAMES = 3
MIN_SCORE = 0.86
OCR_POLL_S = 0.15
MAX_OCR_SECONDS = 6.0

# ----------------------------
# Movement: small circle
# ----------------------------
MOVE_TAP_S = 0.10
MOVE_GAP_S = 0.02

# ----------------------------
# Files
# ----------------------------
POKEMON_FILE = Path("pokemon.txt")
TARGETS_FILE = Path("targets.txt")

# ----------------------------
# Global state (hotkeys)
# ----------------------------
stop_requested = False
paused = False
found_target: str | None = None
open_picker_requested = False

state_lock = threading.Lock()


# ============================
# Helpers
# ============================
def ts() -> str:
    return time.strftime("%H:%M:%S")


def dbg(msg: str):
    print(f"[{ts()}] {msg}")


def rgb_to_bgr(rgb):
    r, g, b = rgb
    return (b, g, r)


def close_bgr(a, b, tol=30) -> bool:
    return (abs(a[0] - b[0]) <= tol and
            abs(a[1] - b[1]) <= tol and
            abs(a[2] - b[2]) <= tol)


def norm_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^A-Za-z]", "", s)
    return s.lower()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def load_list_normalized(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    seen = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        n = norm_name(line)
        if n and n not in seen:
            out.append(n)
            seen.add(n)
    return out


def best_fuzzy_match(ocr: str, whitelist: list[str], min_score: float) -> tuple[str | None, float]:
    if not ocr:
        return None, 0.0

    L = len(ocr)
    candidates = whitelist
    if L <= 5:
        candidates = [w for w in whitelist if abs(len(w) - L) <= 1]

    best = None
    best_sc = 0.0
    for w in candidates:
        sc = similarity(ocr, w)
        if sc > best_sc:
            best_sc = sc
            best = w

    if best is None or best_sc < min_score:
        return None, best_sc
    return best, best_sc


def pause_point():
    while True:
        with state_lock:
            if stop_requested:
                return
            if not paused:
                return
        time.sleep(0.05)


# ============================
# Window helpers
# ============================
def find_roblox_hwnd():
    target_exes = {"RobloxPlayerBeta.exe", "RobloxStudioBeta.exe"}
    candidates = []

    def enum_cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        if win32gui.IsIconic(hwnd):
            return
        if not win32gui.GetWindowText(hwnd).strip():
            return
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            exe = psutil.Process(pid).name()
        except Exception:
            return
        if exe in target_exes:
            try:
                l, t, r, b = win32gui.GetClientRect(hwnd)
                area = (r - l) * (b - t)
            except Exception:
                area = 0
            candidates.append((area, hwnd))

    win32gui.EnumWindows(enum_cb, None)
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def set_window_rect(hwnd, x, y, w, h):
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, x, y, w, h, win32con.SWP_SHOWWINDOW)


def activate_window(hwnd):
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
    time.sleep(0.02)


def get_client_rect_on_screen(hwnd):
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    w = right - left
    h = bottom - top
    sx, sy = win32gui.ClientToScreen(hwnd, (0, 0))
    return sx, sy, w, h


def client_to_screen(hwnd, cx, cy):
    return win32gui.ClientToScreen(hwnd, (int(cx), int(cy)))


def click_client(hwnd, cx, cy, post_sleep=0.10):
    pause_point()
    activate_window(hwnd)
    sx, sy = client_to_screen(hwnd, cx, cy)
    pdi.moveTo(sx, sy, duration=0.05)
    pdi.mouseDown()
    time.sleep(0.03)
    pdi.mouseUp()
    time.sleep(post_sleep)


# ============================
# Capture helpers
# ============================
def capture_client_bgr(hwnd, sct):
    pause_point()
    x, y, w, h = get_client_rect_on_screen(hwnd)
    shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
    return np.array(shot)[:, :, :3]  # BGR


def avg_patch_bgr(frame, x, y, r=2):
    h, w = frame.shape[:2]
    x0 = max(0, x - r)
    x1 = min(w - 1, x + r)
    y0 = max(0, y - r)
    y1 = min(h - 1, y + r)
    patch = frame[y0:y1 + 1, x0:x1 + 1]
    bgr = patch.reshape(-1, 3).mean(axis=0)
    return tuple(int(round(v)) for v in bgr.tolist())


def fight_visible(frame_bgr) -> bool:
    got = avg_patch_bgr(frame_bgr, FIGHT_ANCHOR_XY[0], FIGHT_ANCHOR_XY[1], r=2)
    expected = rgb_to_bgr(FIGHT_EXPECTED_RGB)
    return close_bgr(got, expected, tol=FIGHT_TOL)


# ============================
# OCR
# ============================
def crop_norm(frame_bgr, roi):
    h, w = frame_bgr.shape[:2]
    l, t, r, b = roi
    x1 = int(w * l)
    y1 = int(h * t)
    x2 = int(w * r)
    y2 = int(h * b)
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    return frame_bgr[y1:y2, x1:x2].copy()


def preprocess_for_name(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    return th


def ocr_name(frame_bgr) -> str:
    roi = crop_norm(frame_bgr, POKENAME_ROI)
    th = preprocess_for_name(roi)
    raw = pytesseract.image_to_string(th, config=TESS_CONFIG)
    return norm_name(raw)


# ============================
# Alerts + Movement
# ============================
def beep_pattern():
    for _ in range(5):
        winsound.Beep(1100, 140)
        time.sleep(0.06)


def circle_step():
    for key in ("w", "d", "s", "a"):
        pause_point()
        pdi.keyDown(key)
        time.sleep(MOVE_TAP_S)
        pdi.keyUp(key)
        time.sleep(MOVE_GAP_S)


# ============================
# Picker UI (Tkinter)
# ============================
def run_picker_and_write_targets(all_names: list[str], targets_path: Path) -> list[str]:
    import tkinter as tk
    from tkinter import ttk, messagebox

    def dedupe_casefold(seq: list[str]) -> list[str]:
        seen = set()
        out = []
        for s in seq:
            k = s.casefold()
            if k not in seen:
                out.append(s)
                seen.add(k)
        return out

    class Picker(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Pokémon Targets")
            self.geometry("720x420")
            self.minsize(640, 380)

            self.query_var = tk.StringVar()
            self.status_var = tk.StringVar(value="Type to search. Double-click or Enter to add. Start saves and continues.")
            self.filtered = all_names[:]

            self._build()
            self._bind()
            self._refresh()

            if targets_path.exists():
                existing = [ln.strip() for ln in targets_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
                for n in existing:
                    self._add_target(n, quiet=True)

        def _build(self):
            top = ttk.Frame(self, padding=10)
            top.pack(fill="both", expand=True)

            row1 = ttk.Frame(top)
            row1.pack(fill="x")

            ttk.Label(row1, text="Search:").pack(side="left")
            self.entry = ttk.Entry(row1, textvariable=self.query_var)
            self.entry.pack(side="left", fill="x", expand=True, padx=(8, 8))

            self.btn_add = ttk.Button(row1, text="+ Add", command=self.add_current)
            self.btn_add.pack(side="left")

            mid = ttk.Frame(top)
            mid.pack(fill="both", expand=True, pady=(10, 10))

            left = ttk.Frame(mid)
            left.pack(side="left", fill="both", expand=True)
            right = ttk.Frame(mid)
            right.pack(side="left", fill="both", expand=True, padx=(10, 0))

            ttk.Label(left, text="Matches").pack(anchor="w")
            self.listbox = tk.Listbox(left, height=12, activestyle="dotbox")
            self.listbox.pack(fill="both", expand=True)

            ttk.Label(right, text="Selected").pack(anchor="w")
            self.selected = tk.Listbox(right, height=12, activestyle="dotbox")
            self.selected.pack(fill="both", expand=True)

            row2 = ttk.Frame(top)
            row2.pack(fill="x")

            self.btn_remove = ttk.Button(row2, text="Remove", command=self.remove_selected)
            self.btn_clear = ttk.Button(row2, text="Clear", command=self.clear_all)
            self.btn_start = ttk.Button(row2, text="Start (Save)", command=self.save_and_close)

            self.btn_remove.pack(side="left")
            self.btn_clear.pack(side="left", padx=(8, 0))
            self.btn_start.pack(side="right")

            ttk.Label(top, textvariable=self.status_var).pack(fill="x")

        def _bind(self):
            self.entry.bind("<KeyRelease>", lambda e: self.on_change())
            self.entry.bind("<Return>", lambda e: self.add_current())
            self.entry.bind("<Down>", lambda e: self._focus_matches())
            self.listbox.bind("<Double-Button-1>", lambda e: self.add_current())
            self.listbox.bind("<Return>", lambda e: self.add_current())

        def _focus_matches(self):
            if self.listbox.size() > 0:
                self.listbox.focus_set()
            return "break"

        def on_change(self):
            q = self.query_var.get().strip().casefold()
            if not q:
                self.filtered = all_names[:]
            else:
                self.filtered = [n for n in all_names if q in n.casefold()]
            self._refresh()

        def _refresh(self):
            self.listbox.delete(0, win32con.MAX_PATH)  # safe clear
            self.listbox.delete(0, "end")
            for n in self.filtered[:500]:
                self.listbox.insert("end", n)
            if self.filtered:
                self.listbox.selection_clear(0, "end")
                self.listbox.selection_set(0)
                self.listbox.activate(0)
            self._status()

        def _status(self):
            self.status_var.set(f"Matches: {len(self.filtered)} (showing 500) | Selected: {self.selected.size()}")

        def _get_current_match(self) -> str | None:
            if self.listbox.size() == 0:
                return None
            sel = self.listbox.curselection()
            idx = sel[0] if sel else 0
            return self.listbox.get(idx)

        def _add_target(self, name: str, quiet: bool = False):
            name = name.strip()
            if not name:
                return
            existing = [self.selected.get(i) for i in range(self.selected.size())]
            if any(name.casefold() == e.casefold() for e in existing):
                if not quiet:
                    self.status_var.set(f"Already selected: {name}")
                return
            self.selected.insert("end", name)
            if not quiet:
                self.status_var.set(f"Added: {name}")

        def add_current(self):
            n = self._get_current_match()
            if n:
                self._add_target(n)
            self.entry.focus_set()

        def remove_selected(self):
            sel = self.selected.curselection()
            if not sel:
                return
            for idx in reversed(sel):
                self.selected.delete(idx)
            self._status()

        def clear_all(self):
            self.selected.delete(0, "end")
            self._status()

        def save_and_close(self):
            items = [self.selected.get(i).strip() for i in range(self.selected.size())]
            items = [x for x in items if x]
            items = dedupe_casefold(items)

            if not items:
                messagebox.showwarning("No targets", "Add at least one Pokémon before starting.")
                return

            targets_path.write_text("\n".join(items) + "\n", encoding="utf-8")
            messagebox.showinfo("Saved", f"Saved {len(items)} target(s) to {targets_path.name}")
            self.destroy()

    app = Picker()
    app.mainloop()

    if not targets_path.exists():
        return []
    return [ln.strip() for ln in targets_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]


# ============================
# Overlay bar (click-through, translucent)
# Uses win32ui for font selection (win32gui.CreateFont does not exist)
# ============================
class OverlayBar:
    def __init__(self, alpha: int = 110, height: int = 34, width: int = 980, top_margin: int = 10):
        self.alpha = max(30, min(255, alpha))
        self.height = height
        self.width = width
        self.top_margin = top_margin

        self.hwnd = None
        self._class_name = "OCRPokemonOverlayBar"
        self._text = "Starting..."
        self._bg_rgb = (20, 20, 20)   # darker than pure black so it looks nicer
        self._fg_rgb = (255, 255, 255)

        self._wndproc_ref = self._wndproc

        self._register_class()
        self._create_window()

    def _register_class(self):
        wc = win32gui.WNDCLASS()
        wc.hInstance = win32api.GetModuleHandle(None)
        wc.lpszClassName = self._class_name
        wc.lpfnWndProc = self._wndproc_ref
        wc.hCursor = win32gui.LoadCursor(None, win32con.IDC_ARROW)
        wc.hbrBackground = win32con.COLOR_WINDOW
        try:
            win32gui.RegisterClass(wc)
        except Exception:
            pass

    def _create_window(self):
        ex_style = (
            win32con.WS_EX_TOPMOST |
            win32con.WS_EX_TOOLWINDOW |
            win32con.WS_EX_LAYERED |
            win32con.WS_EX_TRANSPARENT |
            win32con.WS_EX_NOACTIVATE
        )
        style = win32con.WS_POPUP

        self.hwnd = win32gui.CreateWindowEx(
            ex_style,
            self._class_name,
            "",
            style,
            0, 0, self.width, self.height,
            0, 0, win32api.GetModuleHandle(None), None
        )

        win32gui.SetLayeredWindowAttributes(self.hwnd, 0, self.alpha, win32con.LWA_ALPHA)
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
        win32gui.UpdateWindow(self.hwnd)

    def destroy(self):
        if self.hwnd:
            try:
                win32gui.DestroyWindow(self.hwnd)
            except Exception:
                pass
            self.hwnd = None

    def set_text(self, text: str):
        self._text = text
        if self.hwnd:
            win32gui.InvalidateRect(self.hwnd, None, True)

    def set_position_over_client(self, roblox_hwnd: int):
        if not self.hwnd:
            return
        sx, sy, cw, ch = get_client_rect_on_screen(roblox_hwnd)
        w = min(self.width, cw - 20) if cw > 40 else self.width
        h = self.height

        x = sx + (cw - w) // 2
        y = sy + self.top_margin

        win32gui.SetWindowPos(
            self.hwnd,
            win32con.HWND_TOPMOST,
            int(x), int(y), int(w), int(h),
            win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW
        )

    def pump(self):
        if not self.hwnd:
            return
        try:
            win32gui.PumpWaitingMessages()
        except Exception:
            pass

    def _wndproc(self, hwnd, msg, wparam, lparam):
        try:
            if msg == win32con.WM_PAINT:
                self._on_paint(hwnd)
                return 0
            if msg == win32con.WM_ERASEBKGND:
                return 1
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)
        except Exception:
            # swallow paint errors so we don't spam
            return 0

    def _on_paint(self, hwnd):
        hdc, ps = win32gui.BeginPaint(hwnd)
        try:
            rc = win32gui.GetClientRect(hwnd)

            # background
            brush = win32gui.CreateSolidBrush(win32api.RGB(*self._bg_rgb))
            win32gui.FillRect(hdc, rc, brush)
            win32gui.DeleteObject(brush)

            # text
            win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
            win32gui.SetTextColor(hdc, win32api.RGB(*self._fg_rgb))

            left, top, right, bottom = rc
            text_rc = (left + 12, top + 7, right - 12, bottom - 7)

            # Select a font using win32ui (this is the correct API)
            dc = win32ui.CreateDCFromHandle(hdc)

            font = win32ui.CreateFont({
                "name": "Segoe UI",
                "height": 18,         # px-ish, looks right at 1080p scaling too
                "weight": 600,        # semi-bold
                "quality": win32con.CLEARTYPE_QUALITY,
            })

            old = dc.SelectObject(font)
            win32gui.DrawText(
                hdc,
                self._text,
                -1,
                text_rc,
                win32con.DT_LEFT | win32con.DT_VCENTER | win32con.DT_SINGLELINE | win32con.DT_END_ELLIPSIS
            )
            dc.SelectObject(old)

        finally:
            win32gui.EndPaint(hwnd, ps)


# ============================
# Hotkeys
# ============================
def on_key_press(key):
    global stop_requested, paused, found_target, open_picker_requested

    if key == keyboard.Key.esc:
        with state_lock:
            stop_requested = True
        return False

    if isinstance(key, keyboard.KeyCode) and key.char:
        c = key.char.lower()
        if c == "p":
            with state_lock:
                paused = not paused
            dbg("HOTKEY: P toggled pause")
            winsound.Beep(800 if paused else 1200, 120)

        elif c == "r":
            with state_lock:
                found_target = None
            dbg("HOTKEY: R resume after catch (cleared found_target)")
            winsound.Beep(1200, 120)

        elif c == "o":
            with state_lock:
                open_picker_requested = True
            dbg("HOTKEY: O open picker requested")
            winsound.Beep(1000, 120)


# ============================
# Main
# ============================
def main():
    global stop_requested, paused, found_target, open_picker_requested

    if not POKEMON_FILE.exists():
        dbg(f"Missing {POKEMON_FILE.name} (one name per line)")
        return

    all_names_display = [ln.strip() for ln in POKEMON_FILE.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    if not all_names_display:
        dbg(f"{POKEMON_FILE.name} is empty")
        return

    # Picker first
    picked = run_picker_and_write_targets(all_names_display, TARGETS_FILE)
    if not picked:
        dbg("No targets selected. Exiting.")
        return

    whitelist = load_list_normalized(POKEMON_FILE)

    def load_targets_norm() -> set[str]:
        if not TARGETS_FILE.exists():
            return set()
        names = [ln.strip() for ln in TARGETS_FILE.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        return {norm_name(x) for x in names if norm_name(x)}

    targets_norm = load_targets_norm()
    dbg(f"Targets: {', '.join(sorted(targets_norm))}")

    hwnd = find_roblox_hwnd()
    if not hwnd:
        dbg("Roblox window not found")
        return

    set_window_rect(hwnd, WIN_X, WIN_Y, WIN_W, WIN_H)
    activate_window(hwnd)
    time.sleep(0.2)

    overlay = OverlayBar(alpha=110, height=34, width=980, top_margin=12)
    overlay.set_position_over_client(hwnd)

    kb = keyboard.Listener(on_press=on_key_press)
    kb.start()

    dbg("Running. Hotkeys: ESC stop | P pause | R resume after catch | O picker")

    next_beep = 0.0
    last_overlay_update = 0.0

    try:
        with mss.mss() as sct:
            while True:
                pause_point()

                with state_lock:
                    if stop_requested:
                        break
                    lp = paused
                    lf = found_target
                    lop = open_picker_requested

                tnow = time.time()
                if tnow - last_overlay_update >= 0.20:
                    if lp:
                        status = "PAUSED"
                    elif lf:
                        status = f"FOUND {lf.upper()} (press R after catch)"
                    else:
                        status = "RUNNING"

                    hotkeys = "ESC Stop  |  P Pause  |  R Resume after catch  |  O Pick targets"
                    overlay.set_text(f"{status}    {hotkeys}")
                    overlay.set_position_over_client(hwnd)
                    last_overlay_update = tnow

                overlay.pump()

                if lop:
                    with state_lock:
                        open_picker_requested = False
                        paused = True
                    overlay.set_text("PICKING TARGETS    Close picker to continue")
                    overlay.set_position_over_client(hwnd)

                    run_picker_and_write_targets(all_names_display, TARGETS_FILE)
                    targets_norm = load_targets_norm()
                    dbg(f"Targets reloaded: {', '.join(sorted(targets_norm))}")

                    with state_lock:
                        paused = False
                    continue

                if lf is not None:
                    if tnow >= next_beep:
                        dbg(f"ALERT: target '{lf}'")
                        beep_pattern()
                        next_beep = time.time() + 5.0
                    time.sleep(0.05)
                    continue

                circle_step()

                frame = capture_client_bgr(hwnd, sct)
                if not fight_visible(frame):
                    time.sleep(ENCOUNTER_POLL_S)
                    continue

                dbg("Encounter detected. OCR...")

                last_match = None
                streak = 0
                start = time.time()

                while (time.time() - start) < MAX_OCR_SECONDS:
                    pause_point()
                    with state_lock:
                        if stop_requested:
                            break
                        lp2 = paused
                    if lp2:
                        time.sleep(0.05)
                        continue

                    frame = capture_client_bgr(hwnd, sct)
                    raw = ocr_name(frame)
                    matched, score = best_fuzzy_match(raw, whitelist, MIN_SCORE)

                    if matched is None:
                        last_match = None
                        streak = 0
                    else:
                        if matched == last_match:
                            streak += 1
                        else:
                            last_match = matched
                            streak = 1

                        if streak >= STABLE_FRAMES:
                            if matched in targets_norm:
                                with state_lock:
                                    found_target = matched
                                dbg(f"FOUND TARGET: {matched} (press R after catch to continue)")
                                next_beep = 0.0
                                break
                            else:
                                dbg(f"Not target: {matched}. RUN.")
                                click_client(hwnd, RUN_CLICK[0], RUN_CLICK[1], post_sleep=0.25)
                                break

                    time.sleep(OCR_POLL_S)

                time.sleep(0.35)

    except KeyboardInterrupt:
        dbg("Stopped by Ctrl+C")
    finally:
        overlay.destroy()
        dbg("Exited cleanly")


if __name__ == "__main__":
    main()
