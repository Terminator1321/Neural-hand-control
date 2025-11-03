import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread
from PIL import Image, ImageTk
import math, random

from Gesture_Mouse_Controller import GestureController, CONFIG
from Gesture.Gesture_GUI import start_visualizer

class AnimatedBackground(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.particles = []
        self.width = 0
        self.height = 0
        self.bind("<Configure>", self.on_resize)
        self.after(100, self.initialize)

    def initialize(self):
        self.width = self.winfo_width()
        self.height = self.winfo_height()
        if self.width > 1 and self.height > 1:
            self.create_particles()
            self.animate()

    def create_particles(self):
        self.particles.clear()
        w, h = self.width, self.height
        count = max(50, min(100, (w * h) // 12000))
        for _ in range(count):
            self.particles.append({
                'x': random.uniform(0, w),
                'y': random.uniform(0, h),
                'vx': random.uniform(-0.8, 0.8),
                'vy': random.uniform(-0.8, 0.8),
                'glow': random.uniform(0, 2 * math.pi),
                'size': random.uniform(1.5, 3)
            })

    def on_resize(self, event):
        self.width = event.width
        self.height = event.height
        if not self.particles and self.width > 1 and self.height > 1:
            self.create_particles()

    def animate(self):
        self.delete("all")
        w, h = self.width, self.height

        for i, p1 in enumerate(self.particles):
            for p2 in self.particles[i+1:]:
                d = math.hypot(p1['x']-p2['x'], p1['y']-p2['y'])
                if d < 150:
                    alpha = int(255 * (1 - d/150))
                    color = f"#{alpha//3:02x}{alpha//4:02x}{alpha:02x}"
                    self.create_line(p1['x'], p1['y'], p2['x'], p2['y'], 
                                   fill=color, width=1, tags="line")

        for p in self.particles:
            p['x'] = (p['x'] + p['vx']) % w
            p['y'] = (p['y'] + p['vy']) % h
            p['glow'] += 0.08
            glow = int(180 + 75 * math.sin(p['glow']))
            color = f"#{glow:02x}{glow//2:02x}{255:02x}"
            size = p['size']
            self.create_oval(p['x']-size, p['y']-size, p['x']+size, p['y']+size, fill=color, outline=color, tags="particle")
        
        self.after(33, self.animate)


class TranslucentButton(tk.Canvas):
    def __init__(self, parent, text, command, color="#00ffff", **kwargs):
        super().__init__(parent, highlightthickness=0, **kwargs)
        self.text = text
        self.command = command
        self.color = color
        self.enabled = True
        self.phase = 0
        self.hovered = False
        
        self.bind("<Button-1>", self.on_click)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Configure>", lambda e: self.draw())
        
        self.animate()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10:
            return
        
        base = tuple(int(self.color[i:i+2], 16) for i in (1, 3, 5))
        
        intensity = int(30 + 30 * math.sin(self.phase))
        glow = tuple(min(c + intensity, 255) for c in base)
        glow_color = f"#{glow[0]:02x}{glow[1]:02x}{glow[2]:02x}"
        
        if self.hovered:
            bg_opacity = 60
        else:
            bg_opacity = 30
        
        bg_color = f"#{base[0]//8:02x}{base[1]//8:02x}{base[2]//8:02x}"
        
        self.create_rectangle(3, 3, w-3, h-3, fill=bg_color, outline="", tags="bg")
        
        self.create_rectangle(3, 3, w-3, h-3, outline=glow_color, width=2, tags="border")
        
        text_color = "white" if self.enabled else "#666666"
        self.create_text(w//2, h//2, text=self.text, fill=text_color, font=("Orbitron", 12, "bold"), tags="text")

    def animate(self):
        self.phase += 0.12
        self.draw()
        self.after(50, self.animate)

    def on_enter(self, e):
        self.hovered = True
        self.config(cursor="hand2")
        self.draw()

    def on_leave(self, e):
        self.hovered = False
        self.config(cursor="")
        self.draw()

    def on_click(self, e):
        if self.enabled and self.command:
            self.command()

    def set_enabled(self, state):
        self.enabled = state
        self.draw()


class SplashScreen:
    def __init__(self, root):
        self.root = root
        self.top = tk.Toplevel(root)
        self.top.overrideredirect(True)
        
        splash_width = 800
        splash_height = 600
        screen_width = self.top.winfo_screenwidth()
        screen_height = self.top.winfo_screenheight()
        x = (screen_width - splash_width) // 2
        y = (screen_height - splash_height) // 2
        
        self.top.geometry(f"{splash_width}x{splash_height}+{x}+{y}")
        self.top.configure(bg="#0a0e27")
        
        try:
            bg_image = Image.open("Assets/bg.png")
            bg_image = bg_image.resize((splash_width, splash_height), Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            
            canvas = tk.Canvas(self.top, width=splash_width, height=splash_height, highlightthickness=0, bg="#0a0e27")
            canvas.pack(fill="both", expand=True)
            
            canvas.create_image(0, 0, anchor="nw", image=self.bg_photo)
            
            canvas.create_rectangle(0, splash_height//2 - 100, splash_width, splash_height//2 + 100, fill="#0a0e27", stipple="gray50", outline="")
            
            canvas.create_text(splash_width//2, splash_height//2 - 30, text="HAND GESTURE CONTROL", font=("Orbitron", 32, "bold"), fill="#00ffff")
            
            canvas.create_text(splash_width//2, splash_height//2 + 30, text="Loading Neural Core...", font=("Consolas", 14), fill="#00ff88")
            
        except FileNotFoundError:
            bg = AnimatedBackground(self.top, bg="#0a0e27")
            bg.pack(fill="both", expand=True)
            
            overlay = tk.Frame(self.top, bg="#0a0e27")
            overlay.place(relx=0.5, rely=0.5, anchor="center")
            
            tk.Label(overlay, text="HAND GESTURE CONTROL", font=("Orbitron", 32, "bold"), fg="#00ffff", bg="#0a0e27").pack(pady=20)
            tk.Label(overlay, text="Loading Neural Core...", font=("Consolas", 14), fg="#00ff88", bg="#0a0e27").pack()
            
            print("⚠ Warning: assets/bg.png not found, using animated background")
        
        self.top.after(3000, self.top.destroy)

class HandGestureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Neural Hand Control Interface")
        self.root.geometry("1000x750")
        self.root.configure(bg="#0a0e27")
        
        self.center_window()

        self.is_running = False
        self.controller = None
        self.thread = None

        self.scroll_speed = tk.DoubleVar(value=10)
        self.mouse_speed = tk.DoubleVar(value=1200)
        self.smoothing_factor = tk.DoubleVar(value=0.25)
        self.mode = tk.StringVar(value="trackpad")
        self.brightness_hand = tk.StringVar(value="Left")
        self.volume_hand = tk.StringVar(value="Right")
        self.mode = tk.StringVar(value="trackpad")

        self.mode.trace_add("write", self.on_mode_change)
        self.brightness_hand.trace_add("write", self.on_brightness_hand_change)
        self.volume_hand.trace_add("write", self.on_volume_hand_change)

        self.build_ui()

    def center_window(self):
        self.root.update_idletasks()
        width = 1000
        height = 750
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def on_brightness_hand_change(self, *args):
        """When brightness hand changes, set volume hand to opposite"""
        if self.brightness_hand.get() == "Left":
            self.volume_hand.set("Right")
        else:
            self.volume_hand.set("Left")

    def on_volume_hand_change(self, *args):
        """When volume hand changes, set brightness hand to opposite"""
        if self.volume_hand.get() == "Left":
            self.brightness_hand.set("Right")
        else:
            self.brightness_hand.set("Left")
    
    def on_mode_change(self, *args):
        CONFIG.default_mode = self.mode.get()


    def start_controller(self):
        if self.is_running:
            return
        CONFIG.scroll_speed = self.scroll_speed.get()
        CONFIG.mouse_speed = self.mouse_speed.get()
        CONFIG.smoothing_factor = self.smoothing_factor.get()
        CONFIG.default_mode = self.mode.get()
        CONFIG.brightness_hand_label = self.brightness_hand.get()
        CONFIG.volume_hand_label = self.volume_hand.get()
        self.controller = GestureController(CONFIG)
        self.is_running = True
        self.start_btn.set_enabled(False)
        self.stop_btn.set_enabled(True)
        Thread(target=lambda: self.controller.run(show=False), daemon=True).start()
        self.update_status("STATUS: ACTIVE", "#00ffff")
        messagebox.showinfo("Neural Link", "✓ Gesture controller activated!")

    def stop_controller(self):
        if not self.is_running:
            return
        self.is_running = False
        self.start_btn.set_enabled(True)
        self.stop_btn.set_enabled(False)
        if self.controller:
            self.controller.stop()
        self.update_status("STATUS: DISCONNECTED", "#ff0066")
        messagebox.showinfo("Neural Link", "✓ Controller stopped.")

    def save_settings(self):
        CONFIG.scroll_speed = self.scroll_speed.get()
        CONFIG.mouse_speed = self.mouse_speed.get()
        CONFIG.smoothing_factor = self.smoothing_factor.get()
        CONFIG.default_mode = self.mode.get()
        CONFIG.brightness_hand_label = self.brightness_hand.get()
        CONFIG.volume_hand_label = self.volume_hand.get()
        messagebox.showinfo("Neural Core", "✓ Settings saved successfully!")

    def check_gesture(self):
        Thread(target=start_visualizer, daemon=True).start()

    def update_status(self, text, color):
        self.status.itemconfig(self.status_text, text=text, fill=color)

    def build_ui(self):
        self.bg = AnimatedBackground(self.root, bg="#0a0e27")
        self.bg.place(relx=0, rely=0, relwidth=1, relheight=1)

        title_label = tk.Label(self.root, text="⚡ NEURAL HAND CONTROL ⚡",font=("Orbitron", 30, "bold"), fg="#00ffff", bg="#0a0e27")
        title_label.place(relx=0.5, rely=0.06, anchor="center")

        settings_frame = tk.Frame(self.root, bg="#1a1f3a")
        settings_frame.place(relx=0.1, rely=0.15, relwidth=0.8, relheight=0.5)

        tk.Label(settings_frame, text="⚙ CONFIGURATION PANEL", font=("Orbitron", 16, "bold"), fg="#00ffff", bg="#1a1f3a").place(relx=0.5, rely=0.06, anchor="center")

        tk.Label(settings_frame, text="Scroll Speed", font=("Consolas", 11), fg="white", bg="#1a1f3a").place(relx=0.05, rely=0.25)
        scroll_slider = tk.Scale(settings_frame, from_=1, to=50, variable=self.scroll_speed, orient="horizontal",bg="#2a2f4a", fg="white", troughcolor="#1a1f3a", highlightthickness=0, length=250)
        scroll_slider.place(relx=0.25, rely=0.24)

        tk.Label(settings_frame, text="Mouse Speed", font=("Consolas", 11), fg="white", bg="#1a1f3a").place(relx=0.05, rely=0.45)
        mouse_slider = tk.Scale(settings_frame, from_=200, to=3000, variable=self.mouse_speed, orient="horizontal",bg="#2a2f4a", fg="white", troughcolor="#1a1f3a", highlightthickness=0, length=250)
        mouse_slider.place(relx=0.25, rely=0.44)

        tk.Label(settings_frame, text="Smoothing", font=("Consolas", 11), fg="white", bg="#1a1f3a").place(relx=0.05, rely=0.65)
        smooth_slider = tk.Scale(settings_frame, from_=0.1, to=1.0, resolution=0.01, variable=self.smoothing_factor, orient="horizontal", bg="#2a2f4a", fg="white", troughcolor="#1a1f3a", highlightthickness=0, length=250)
        smooth_slider.place(relx=0.25, rely=0.64)

        tk.Label(settings_frame, text="Control Mode", font=("Consolas", 11), fg="white", bg="#1a1f3a").place(relx=0.55, rely=0.25)
        mode_combo = ttk.Combobox(settings_frame, textvariable=self.mode, values=["trackpad", "joystick"], state="readonly", width=20)
        mode_combo.place(relx=0.75, rely=0.25)

        tk.Label(settings_frame, text="Brightness Hand", font=("Consolas", 11), fg="white", bg="#1a1f3a").place(relx=0.55, rely=0.45)
        brightness_hand_combo = ttk.Combobox(settings_frame, textvariable=self.brightness_hand, values=["Left", "Right"], state="readonly", width=20)
        brightness_hand_combo.place(relx=0.75, rely=0.45)

        tk.Label(settings_frame, text="Volume Hand", font=("Consolas", 11), fg="white", bg="#1a1f3a").place(relx=0.55, rely=0.65)
        volume_hand_combo = ttk.Combobox(settings_frame, textvariable=self.volume_hand, values=["Left", "Right"], state="readonly", width=20)
        volume_hand_combo.place(relx=0.75, rely=0.65)

        btn_container = tk.Frame(self.root, bg="#0a0e27")
        btn_container.place(relx=0.1, rely=0.68, relwidth=0.8, relheight=0.22)

        self.start_btn = TranslucentButton(btn_container, "▶ START", self.start_controller, "#00ff00", bg="#0a0e27")
        self.start_btn.place(relx=0.02, rely=0.1, relwidth=0.28, relheight=0.35)

        self.stop_btn = TranslucentButton(btn_container, "⏹ STOP", self.stop_controller, "#ff0066", bg="#0a0e27")
        self.stop_btn.place(relx=0.36, rely=0.1, relwidth=0.28, relheight=0.35)
        self.stop_btn.set_enabled(False)

        self.save_btn = TranslucentButton(btn_container, "💾 SAVE", self.save_settings, "#0088ff", bg="#0a0e27")
        self.save_btn.place(relx=0.70, rely=0.1, relwidth=0.28, relheight=0.35)
        self.check_btn = TranslucentButton(btn_container, "👋 CHECK GESTURE", self.check_gesture, "#ff00ff", bg="#0a0e27")
        self.check_btn.place(relx=0.02, rely=0.55, relwidth=0.96, relheight=0.35)
        self.status = tk.Canvas(self.root, height=40, bg="#0f1629", highlightthickness=0)
        self.status.place(relx=0, rely=0.93, relwidth=1, relheight=0.07)
        self.status_text = self.status.create_text(500, 20, text="STATUS: READY", fill="#00ff88", font=("Consolas", 13, "bold"))

def main():
    root = tk.Tk()
    root.withdraw()
    SplashScreen(root)
    root.after(3100, lambda: [root.deiconify(), HandGestureGUI(root)])
    root.mainloop()


if __name__ == "__main__":
    main()