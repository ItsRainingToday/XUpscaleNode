from __future__ import annotations

import asyncio
import logging
import queue
import threading
import tkinter as tk

import customtkinter as ctk

from . import main as node_main
from .config import Config

log = logging.getLogger("xupscale.gui")

APP_VERSION = "1.0"

ACCENT = "#2f7bf6"
ACCENT_HOVER = "#2666d1"
DANGER = "#e5484d"
DANGER_HOVER = "#c23a3e"

_ENGINE_LABELS = {"auto": "Авто", "performance": "Performance", "fidelity": "Fidelity"}
_ENGINE_VALUES = {v: k for k, v in _ENGINE_LABELS.items()}

_RESUME_LABELS = {False: "Продолжить", True: "Сначала"}
_RESUME_VALUES = {v: k for k, v in _RESUME_LABELS.items()}


class QueueLogHandler(logging.Handler):
    """Feeds formatted log lines into a queue.Queue - the GUI drains it on
    the Tk thread via root.after(), since Tk widgets aren't safe to touch
    from the background thread the node's asyncio loop runs on."""

    def __init__(self, line_queue: "queue.Queue[str]"):
        super().__init__()
        self._queue = line_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put_nowait(self.format(record))
        except queue.Full:
            pass


class NodeController:
    """Runs xupscale.main.run() (the same code path --headless uses) in a
    background thread, so the Tk mainloop on the main thread never blocks on
    asyncio. Stop is a plain threading.Event - safe to .set() cross-thread,
    unlike asyncio.Event."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event = threading.Event()
        stop_event = self._stop_event

        def _target() -> None:
            try:
                asyncio.run(node_main.run(self.cfg, stop_event))
            except Exception:
                # An unhandled exception here (e.g. http/tcp server startup
                # failing because something else already holds the port)
                # would otherwise just vanish into Python's default thread
                # excepthook - invisible on a --windowed build with no
                # console/stderr. Routing it through logging instead means
                # it actually reaches the GUI's log panel (see App._start_server).
                logging.getLogger("xupscale.main").exception("node crashed")

        self._thread = threading.Thread(target=_target, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 20.0) -> bool:
        """Returns True once the background thread has actually exited.
        Used to matter: this used to clear self._thread unconditionally
        after the join, regardless of whether it actually finished in time -
        so a slow shutdown (ffmpeg not exiting instantly) made `running`
        report False while the old thread (and its bound ports/mDNS name)
        was still alive, and the GUI would flip back to its idle Start state
        even though the server hadn't stopped - clicking Start again then
        raced the still-dying old instance for the same port."""
        if not self.running:
            return True
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            return False
        self._thread = None
        return True


class App(ctk.CTk):
    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.cfg = Config.load(config_path)
        self.controller = NodeController(self.cfg)
        self._watching_alive = False

        # Building the widgets below (CTkEntry/.set() on segmented buttons
        # and sliders) fires spurious <FocusOut>/command callbacks as focus
        # and layout settle - without this guard, _save() would rewrite
        # config.yaml (dropping its comments) before the window has even
        # finished opening, from events the user never triggered.
        self._ready = False

        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._log_handler = QueueLogHandler(self._log_queue)
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

        self.title("XUpscaleNode")
        self.geometry("1040x680")
        self.minsize(860, 560)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_settings_panel()
        self._build_log_panel()
        self._settings_frame.grid(row=0, column=1, sticky="nsew")

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(150, self._poll_log_queue)
        # A fixed delay, not after_idle: the window manager assigning initial
        # keyboard focus as the window is first mapped fires its own
        # <FocusOut>/<FocusIn> churn on entry widgets around the same time
        # idle callbacks would run - too close a race to rely on ordering.
        self.after(500, self._mark_ready)

        if self.cfg.auto_start:
            self.after(200, self._start_server)

    def _mark_ready(self) -> None:
        self._ready = True

    # ---------------------------------------------------------------- sidebar
    def _build_sidebar(self) -> None:
        bar = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color="gray14")
        bar.grid(row=0, column=0, sticky="ns")
        bar.grid_propagate(False)

        ctk.CTkLabel(bar, text="XUpscale", font=ctk.CTkFont(size=24, weight="bold")).pack(
            anchor="w", padx=24, pady=(24, 0))
        ctk.CTkLabel(bar, text=f"v{APP_VERSION}", text_color="gray60").pack(
            anchor="w", padx=24, pady=(0, 24))

        self._idle_controls = ctk.CTkFrame(bar, fg_color="transparent")
        self._idle_controls.pack(fill="x", padx=24)

        self._port_var = tk.StringVar(value=str(self.cfg.listen_port))
        port_entry = ctk.CTkEntry(self._idle_controls, textvariable=self._port_var)
        port_entry.pack(fill="x", pady=(0, 16))
        port_entry.bind("<FocusOut>", lambda _e: self._on_port_changed())
        port_entry.bind("<Return>", lambda _e: self._on_port_changed())

        self._autostart_var = tk.BooleanVar(value=self.cfg.auto_start)
        ctk.CTkSwitch(self._idle_controls, text="Автозапуск сервера", variable=self._autostart_var,
                      command=self._on_autostart_changed).pack(anchor="w")

        self._running_controls = ctk.CTkFrame(bar, fg_color="transparent")
        row = ctk.CTkFrame(self._running_controls, fg_color="transparent")
        row.pack(fill="x")
        self._address_var = tk.StringVar(value="")
        ctk.CTkEntry(row, textvariable=self._address_var, state="readonly").pack(
            side="left", fill="x", expand=True)
        ctk.CTkButton(row, text="⧉", width=32, command=self._copy_address).pack(
            side="left", padx=(8, 0))
        # _running_controls is only packed into `bar` while the server runs
        # (see _set_running_ui) - it replaces _idle_controls in place.

        self._start_btn = ctk.CTkButton(
            bar, text="▶", width=64, height=64, corner_radius=32,
            font=ctk.CTkFont(size=22), fg_color=ACCENT, hover_color=ACCENT_HOVER,
            command=self._toggle_server,
        )
        self._start_btn.pack(side="bottom", pady=24)

    def _copy_address(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self._address_var.get())

    # ------------------------------------------------------------- settings
    def _build_settings_panel(self) -> None:
        frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        self._settings_frame = frame

        r = 0
        ctk.CTkLabel(frame, text="XUpscale", font=ctk.CTkFont(size=22, weight="bold")).grid(
            row=r, column=0, sticky="w", pady=(4, 20)); r += 1

        self._add_row(frame, r, "Приёмник",
                      "Адрес и порт настоящего XRemotexServer",
                      self._build_receiver_control); r += 1
        self._add_row(frame, r, "Имя узла",
                      "Видно в списке cast-целей приложения",
                      self._build_cast_name_control); r += 1

        self._add_section(frame, r, "Апскейл"); r += 1
        self._add_row(frame, r, "Режим",
                      "Авто выбирает по разрешению источника; Performance/Fidelity - всегда один режим",
                      self._build_engine_control); r += 1
        self._add_row(frame, r, "Качество кодирования",
                      "Шкала 0-51: меньше значение - лучше картинка и выше битрейт",
                      self._build_quality_control); r += 1
        self._add_row(frame, r, "Лимит битрейта",
                      "Потолок среднего битрейта апскейленного потока",
                      self._build_bitrate_control); r += 1

        self._add_section(frame, r, "Просмотр"); r += 1
        self._add_row(frame, r, "Продолжение эпизода",
                      "Что делать при открытии серии, для которой уже есть сохранённая позиция",
                      self._build_resume_control); r += 1

    def _add_section(self, parent: ctk.CTkBaseClass, row: int, title: str) -> None:
        ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(size=17, weight="bold")).grid(
            row=row, column=0, sticky="w", pady=(22, 10))

    def _add_row(self, parent: ctk.CTkBaseClass, row: int, label: str, desc: str, control_builder) -> None:
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.grid(row=row, column=0, sticky="ew", pady=(0, 18))
        wrap.grid_columnconfigure(0, weight=1)

        text_col = ctk.CTkFrame(wrap, fg_color="transparent")
        text_col.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(text_col, text=label, font=ctk.CTkFont(size=14, weight="bold"), anchor="w").pack(anchor="w")
        if desc:
            ctk.CTkLabel(text_col, text=desc, font=ctk.CTkFont(size=11), text_color="gray60",
                        anchor="w", justify="left", wraplength=460).pack(anchor="w")

        control = control_builder(wrap)
        control.grid(row=0, column=1, sticky="e", padx=(20, 4))

    def _build_receiver_control(self, parent: ctk.CTkBaseClass) -> ctk.CTkBaseClass:
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        self._host_var = tk.StringVar(value=self.cfg.receiver_host)
        host_entry = ctk.CTkEntry(wrap, textvariable=self._host_var, width=160, placeholder_text="IP приёмника")
        host_entry.pack(side="left")
        self._rport_var = tk.StringVar(value=str(self.cfg.receiver_port))
        rport_entry = ctk.CTkEntry(wrap, textvariable=self._rport_var, width=70)
        rport_entry.pack(side="left", padx=(8, 0))
        for widget in (host_entry, rport_entry):
            widget.bind("<FocusOut>", lambda _e: self._on_receiver_changed())
            widget.bind("<Return>", lambda _e: self._on_receiver_changed())
        return wrap

    def _on_receiver_changed(self) -> None:
        self.cfg.receiver_host = self._host_var.get().strip() or self.cfg.receiver_host
        try:
            self.cfg.receiver_port = int(self._rport_var.get())
        except ValueError:
            self._rport_var.set(str(self.cfg.receiver_port))
        self._save()

    def _build_cast_name_control(self, parent: ctk.CTkBaseClass) -> ctk.CTkBaseClass:
        self._cast_name_var = tk.StringVar(value=self.cfg.cast_name)
        entry = ctk.CTkEntry(parent, textvariable=self._cast_name_var, width=200)
        entry.bind("<FocusOut>", lambda _e: self._on_cast_name_changed())
        entry.bind("<Return>", lambda _e: self._on_cast_name_changed())
        return entry

    def _on_cast_name_changed(self) -> None:
        self.cfg.cast_name = self._cast_name_var.get().strip() or self.cfg.cast_name
        self._save()

    def _build_engine_control(self, parent: ctk.CTkBaseClass) -> ctk.CTkBaseClass:
        seg = ctk.CTkSegmentedButton(parent, values=list(_ENGINE_LABELS.values()),
                                     command=self._on_engine_changed)
        seg.set(_ENGINE_LABELS.get(self.cfg.default_engine, "Авто"))
        return seg

    def _on_engine_changed(self, choice: str) -> None:
        self.cfg.default_engine = _ENGINE_VALUES.get(choice, "auto")
        self._save()

    def _build_quality_control(self, parent: ctk.CTkBaseClass) -> ctk.CTkBaseClass:
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        self._quality_value_var = tk.StringVar(value=str(self.cfg.encode_quality))
        slider = ctk.CTkSlider(wrap, from_=0, to=51, number_of_steps=51, width=180,
                               command=self._on_quality_changed)
        slider.set(self.cfg.encode_quality)
        slider.pack(side="left")
        ctk.CTkLabel(wrap, textvariable=self._quality_value_var, width=28).pack(side="left", padx=(8, 0))
        return wrap

    def _on_quality_changed(self, value: float) -> None:
        q = round(value)
        self._quality_value_var.set(str(q))
        self.cfg.encode_quality = q
        self._save()

    def _build_bitrate_control(self, parent: ctk.CTkBaseClass) -> ctk.CTkBaseClass:
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        self._bitrate_value_var = tk.StringVar(value=f"{self.cfg.max_bitrate_mbps} Мбит/с")
        slider = ctk.CTkSlider(wrap, from_=5, to=100, number_of_steps=95, width=180,
                               command=self._on_bitrate_changed)
        slider.set(self.cfg.max_bitrate_mbps)
        slider.pack(side="left")
        ctk.CTkLabel(wrap, textvariable=self._bitrate_value_var, width=80).pack(side="left", padx=(8, 0))
        return wrap

    def _on_bitrate_changed(self, value: float) -> None:
        b = round(value)
        self._bitrate_value_var.set(f"{b} Мбит/с")
        self.cfg.max_bitrate_mbps = b
        self._save()

    def _build_resume_control(self, parent: ctk.CTkBaseClass) -> ctk.CTkBaseClass:
        seg = ctk.CTkSegmentedButton(parent, values=list(_RESUME_LABELS.values()),
                                     command=self._on_resume_changed)
        seg.set(_RESUME_LABELS[self.cfg.force_start_from_zero])
        return seg

    def _on_resume_changed(self, choice: str) -> None:
        self.cfg.force_start_from_zero = _RESUME_VALUES.get(choice, False)
        self._save()

    def _on_port_changed(self) -> None:
        try:
            self.cfg.listen_port = int(self._port_var.get())
            self._save()
        except ValueError:
            self._port_var.set(str(self.cfg.listen_port))

    def _on_autostart_changed(self) -> None:
        self.cfg.auto_start = self._autostart_var.get()
        self._save()

    def _save(self) -> None:
        if not self._ready:
            return
        self.cfg.save(self.config_path)

    # ------------------------------------------------------------------ log
    def _build_log_panel(self) -> None:
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        self._log_frame = frame

        toolbar = ctk.CTkFrame(frame, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 8))
        ctk.CTkLabel(toolbar, text="Журнал", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        ctk.CTkButton(toolbar, text="Копировать", width=110, command=self._copy_log).pack(
            side="right", padx=(8, 0))
        ctk.CTkButton(toolbar, text="Очистить", width=110, command=self._clear_log).pack(side="right")

        self._log_box = ctk.CTkTextbox(frame, font=ctk.CTkFont(family="Consolas", size=12), wrap="none")
        self._log_box.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self._log_box.configure(state="disabled")

    def _copy_log(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self._log_box.get("1.0", "end"))

    def _clear_log(self) -> None:
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

    def _poll_log_queue(self) -> None:
        while True:
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._log_box.configure(state="normal")
            self._log_box.insert("end", line + "\n")
            self._log_box.configure(state="disabled")
            self._log_box.see("end")
        self.after(150, self._poll_log_queue)

    # -------------------------------------------------------- start / stop
    def _toggle_server(self) -> None:
        if self.controller.running:
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self) -> None:
        self.cfg.refresh_node_ip()
        logging.getLogger().addHandler(self._log_handler)
        self.controller.start()
        self._set_running_ui(True)
        self._watching_alive = True
        self.after(700, self._watch_alive)

    def _watch_alive(self) -> None:
        # The background thread can die on its own shortly after starting -
        # most commonly the HTTP/TCP ports already being held by something
        # else (e.g. this same node also running --headless via a scheduled
        # task) - and that failure is otherwise invisible on a --windowed
        # build (no console for the default thread excepthook to print to).
        # Without this, controller.running silently goes False, and the next
        # Stop click - reading a server that's already "not running" - would
        # take the Start branch in _toggle_server instead, making Stop look
        # like it just restarts the same failed attempt over and over.
        if not self._watching_alive:
            return
        if not self.controller.running:
            self._watching_alive = False
            log.error("server stopped unexpectedly right after starting - see the error above in this log")
            logging.getLogger().removeHandler(self._log_handler)
            self._set_running_ui(False)
            self._start_btn.configure(state="normal")
            return
        self.after(1000, self._watch_alive)

    def _stop_server(self) -> None:
        self._watching_alive = False
        self._start_btn.configure(state="disabled")

        def _do_stop() -> None:
            stopped = self.controller.stop()
            self.after(0, lambda: self._on_stopped(stopped))

        threading.Thread(target=_do_stop, daemon=True).start()

    def _on_stopped(self, stopped: bool) -> None:
        if stopped:
            logging.getLogger().removeHandler(self._log_handler)
            self._set_running_ui(False)
            self._clear_log()
        else:
            # Still actually running (ffmpeg/shutdown took too long) - leave
            # the UI in its running state so Stop can be retried, instead of
            # lying that it stopped (see NodeController.stop's docstring).
            log.warning("stop timed out - server still appears to be running, leaving UI as-is")
            self._watching_alive = True
            self.after(1000, self._watch_alive)
        self._start_btn.configure(state="normal")

    def _set_running_ui(self, running: bool) -> None:
        if running:
            self._idle_controls.pack_forget()
            self._running_controls.pack(fill="x", padx=24)
            self._address_var.set(f"{self.cfg.node_ip}:{self.cfg.listen_port}")
            self._settings_frame.grid_forget()
            self._log_frame.grid(row=0, column=1, sticky="nsew")
            self._start_btn.configure(text="■", fg_color=DANGER, hover_color=DANGER_HOVER)
        else:
            self._running_controls.pack_forget()
            self._idle_controls.pack(fill="x", padx=24)
            self._log_frame.grid_forget()
            self._settings_frame.grid(row=0, column=1, sticky="nsew")
            self._start_btn.configure(text="▶", fg_color=ACCENT, hover_color=ACCENT_HOVER)

    def _on_close(self) -> None:
        if self.controller.running:
            self.controller.stop()
        self.destroy()


def launch(config_path: str) -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = App(config_path)
    # CTk withdraws the window during its own DPI/theme setup and normally
    # re-shows it on the next idle tick - observed staying withdrawn forever
    # under the frozen --windowed exe (PyInstaller onefile's temp-extraction
    # + re-exec appears to shift that timing enough to miss it, though the
    # window and its mainloop are otherwise alive and responsive). Forcing
    # it here is a harmless no-op when CTk already showed itself normally
    # (e.g. running from source).
    app.deiconify()
    app.lift()
    app.focus_force()
    app.mainloop()
