import tkinter as tk
from tkinter import messagebox, ttk, TclError
import tkinter.font as tkFont
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# A default signal frequency for the sine wave
SIGNAL_FREQUENCY = 1

# --- Algorithm for Quantization ---

def compute_quantized_value(max_range, min_range, bit_rate, analog_voltage):
    """
    Computes the quantized value of a given analog voltage for a single point.
    Clamps the input voltage to the specified range before quantization.
    """
    if max_range <= min_range:
        return min_range
    if bit_rate <= 0:
        raise ValueError("Bit Rate must be a positive integer.")

    num_quantization_levels = 2**bit_rate
    if num_quantization_levels <= 1:
        return min_range

    step_size = (max_range - min_range) / (num_quantization_levels - 1)
    
    clamped_analog_voltage = max(min_range, min(max_range, analog_voltage))
    
    normalized_voltage = (clamped_analog_voltage - min_range) / (max_range - min_range)
    quantized_level_index = round(normalized_voltage * (num_quantization_levels - 1))
    
    quantized_level_index = max(0, min(int(quantized_level_index), num_quantization_levels - 1))
    
    quantized_voltage = min_range + (quantized_level_index * step_size)
    return quantized_voltage

def vectorized_compute_quantized_value(max_range, min_range, bit_rate, analog_signal):
    """
    Computes quantized values for an analog signal array using numpy vectorization.
    """
    if max_range <= min_range:
        return np.full_like(analog_signal, min_range)
    if bit_rate <= 0:
        raise ValueError("Bit Rate must be a positive integer.")

    num_quantization_levels = 2**bit_rate
    if num_quantization_levels <= 1:
        return np.full_like(analog_signal, min_range)

    step_size = (max_range - min_range) / (num_quantization_levels - 1)
    
    clamped_signal = np.clip(analog_signal, min_range, max_range)
    
    normalized_signal = (clamped_signal - min_range) / (max_range - min_range)
    quantized_level_indices = np.round(normalized_signal * (num_quantization_levels - 1))
    quantized_level_indices = np.clip(quantized_level_indices, 0, num_quantization_levels - 1).astype(int)
    
    quantized_voltage = min_range + (quantized_level_indices * step_size)
    return quantized_voltage


# --- GUI Application ---
class QuantizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Analog to Digital Quantizer")

        # --- Define Colors and Fonts ---
        self.bg_color = "#2b2b2b"
        self.frame_bg_color = "#3c3f41"
        self.text_color = "#cccccc"
        self.entry_bg_color = "#454545"
        self.entry_fg_color = "#ffffff"
        self.button_bg_color = "#6a90c9"
        self.button_fg_color = "#ffffff"
        self.button_active_bg_color = "#7da7e3"
        self.quantize_button_color = "#4CAF50"
        self.quantize_button_hover_color = "#66BB6A"
        self.clear_button_color = "#F44336"
        self.clear_button_hover_color = "#E57373"
        self.tooltip_bg = "#FFFFCC"
        self.tooltip_fg = "#333333"
        self.field_tooltip_bg = "#E0FFFF"
        self.field_tooltip_fg = "#333333"
        self.info_button_fg_color = "#FFFF00"
        self.info_button_active_fg_color = "#FFA500"
        self.quantized_value_color = "#FFFF00"
        self.na_color = "#FFFF00"
        self.quantized_graph_output_color = "#9400D3" # DarkViolet (Purple)
        self.title_font = tkFont.Font(family="Arial", size=18, weight="bold", underline=True)
        self.label_font = ("Arial", 11)
        self.entry_font = ("Consolas", 11)
        self.button_font = ("Arial", 10, "bold")
        self.output_label_font = ("Arial", 13, "bold")
        self.output_value_font = ("Consolas", 14, "bold")
        self.tooltip_font = ("Arial", 9)
        self.field_tooltip_font = ("Arial", 9)
        self.italic_font = tkFont.Font(family="Arial", size=11, slant="italic")
        self.plot_bg_color = "#1e1e1e"
        self.analog_line_color = "#32CD32"
        self.quantized_line_color = "#FF0000"
        self.level_line_color = "#DDDDDD"

        self.master.configure(bg=self.bg_color)
        self.master.minsize(800, 600)
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        
        self.master.bind('<Configure>', self._on_window_configure)

        self.entry_fields = {}
        self.field_vars = {}
        self.slider_fields = {}
        self.current_focused_entry = None
        self.tooltip_window = None
        self.field_tooltip_window = None
        self.last_calculated_params = None
        self._plot_update_job = None
        self.fig = None
        self.ax = None
        self.canvas = None
        self.toolbar = None

        self.field_explanations = {
            "Maximum Range": "This defines the highest voltage value that your analog-to-digital converter (ADC) can measure. Setting it too low for a given signal can lead to clipping (distortion) if the signal exceeds this range.",
            "Minimum Range": "This defines the lowest voltage value that your analog-to-digital converter (ADC) can measure. Setting it too high can lead to clipping or loss of lower amplitude information.",
            "Bit Rate": "The 'Bit Rate' (or bit depth) determines the number of quantization levels available. More bits mean finer resolution and less quantization error (noise), improving the quality of the digitized signal.",
            "Analog Voltage": "This is the instantaneous voltage of the analog signal that you want to convert into a digital value. The ADC will approximate this voltage to the nearest available quantization level.",
            "Sample Rate": "The 'Sample Rate' (or sampling frequency) determines how many times per second the analog signal is measured. A higher sample rate provides a more accurate representation of the signal's shape over time."
        }
        
        # --- Main Layout Frames ---
        self.main_frame = tk.Frame(master, padx=20, pady=20, bd=2, relief="groove", bg=self.frame_bg_color, highlightbackground=self.text_color)
        self.main_frame.grid(row=0, column=0, sticky="nsew", pady=20, padx=20)
        self.plot_frame = tk.Frame(master, bg=self.plot_bg_color)
        self.plot_frame.grid(row=0, column=1, sticky="nsew", pady=20, padx=(0, 20))

        # --- Configure column weights for responsiveness ---
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(6, weight=2)

        self.title_label = tk.Label(self.main_frame, text="Analog to Digital Quantizer", font=self.title_font, fg=self.text_color, bg=self.frame_bg_color)
        self.title_label.grid(row=0, column=0, columnspan=7, pady=(15, 20), sticky="ew")
        
        # 'About' button is created and placed at the top-right, but hidden initially.
        self.about_button = ttk.Button(self.main_frame, text="About", command=self._open_about_dialog, style="TButton")
        self.about_button.grid(row=0, column=6, padx=5, pady=5, sticky="ne")
        self.about_button.grid_forget()

        # --- Input Fields ---
        input_specs = [
            ("Maximum Range", "V", 10.0, "double", -20.0, 20.0),
            ("Minimum Range", "V", -10.0, "double", -20.0, 20.0),
            ("Bit Rate", "Bit", 8, "int", 1, 16),
            ("Analog Voltage", "V", 5.0, "double", -10.0, 10.0),
            ("Sample Rate", "Hz", 44100, "int", 1000, 100000)
        ]

        row_num = 1
        for text, unit, default, var_type, s_min, s_max in input_specs:
            self.main_frame.grid_rowconfigure(row_num, weight=1)
            tk.Label(self.main_frame, text=f"{text}:", font=self.label_font, fg=self.text_color, bg=self.frame_bg_color).grid(row=row_num, column=0, sticky="w", pady=5)

            var = tk.IntVar(value=int(default)) if var_type == "int" else tk.DoubleVar(value=default)
            var.trace_add('write', self._schedule_plot_update)
            self.field_vars[text] = var

            entry = tk.Entry(self.main_frame, width=15, bd=1, relief="solid", bg=self.entry_bg_color, fg=self.entry_fg_color, insertbackground=self.entry_fg_color, font=self.entry_font, textvariable=var)
            entry.grid(row=row_num, column=1, pady=5, padx=5, sticky="ew")
            tk.Label(self.main_frame, text=unit, font=self.label_font, fg=self.text_color, bg=self.frame_bg_color).grid(row=row_num, column=2, sticky="w")
            self.entry_fields[text] = entry
            entry.bind("<FocusIn>", self._on_entry_focus_in)

            slider = ttk.Scale(self.main_frame, from_=s_min, to=s_max, orient="horizontal", variable=var, command=lambda val, v=var: v.set(val))
            slider.grid(row=row_num, column=6, padx=5, pady=5, sticky="ew")
            self.slider_fields[text] = slider
            row_num += 1

        # --- Styles ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", background=self.button_bg_color, foreground=self.button_fg_color, font=self.button_font, padding=5, relief="flat")
        style.map("TButton", background=[('active', self.button_active_bg_color)])
        style.configure("Quantize.TButton", background=self.quantize_button_color)
        style.map("Quantize.TButton", background=[('active', self.quantize_button_hover_color)])
        style.configure("Clear.TButton", background=self.clear_button_color)
        style.map("Clear.TButton", background=[('active', self.clear_button_hover_color)])
        style.configure("Small.TButton", font=("Arial", 9, "bold"), width=4, padding=2)
        style.map("Small.TButton", background=[('active', self.button_active_bg_color)])
        style.configure("Info.TButton", background=self.frame_bg_color, foreground=self.info_button_fg_color, font=("Arial", 12, "bold"), width=3, padding=0, relief="flat")
        style.map("Info.TButton", foreground=[('active', self.info_button_active_fg_color)])

        # --- Action Buttons ---
        self.action_buttons_frame = tk.Frame(self.main_frame, bg=self.frame_bg_color)
        self.action_buttons_frame.grid(row=row_num, column=0, columnspan=7, pady=(15, 5), sticky="ew")
        self.action_buttons_frame.grid_columnconfigure((0, 3), weight=1)
        self.quantize_button = ttk.Button(self.action_buttons_frame, text="Quantize", command=self._quantize, width=15, style="Quantize.TButton")
        self.quantize_button.grid(row=0, column=1, padx=5, pady=5)
        self.clear_button = ttk.Button(self.action_buttons_frame, text="Clear", command=self._clear_inputs, width=15, style="Clear.TButton")
        self.clear_button.grid(row=0, column=2, padx=5, pady=5)
        row_num += 1

        # --- Output Section ---
        self.output_frame = tk.Frame(self.main_frame, bg=self.frame_bg_color)
        self.output_frame.grid(row=row_num + 1, column=0, columnspan=7, pady=20, sticky="ew")
        self.output_label_prefix = tk.Label(self.output_frame, text="Quantized Voltage: ", font=self.output_label_font, fg=self.text_color, bg=self.frame_bg_color)
        self.output_label_prefix.pack(side=tk.LEFT)
        self.output_value_text = tk.StringVar(value="N/A")
        self.output_label_value = tk.Label(self.output_frame, textvariable=self.output_value_text, font=self.output_value_font, fg=self.na_color, bg=self.frame_bg_color)
        self.output_label_value.pack(side=tk.LEFT)
        self.output_frame.bind("<Enter>", self._show_output_analysis_tooltip)
        self.output_frame.bind("<Leave>", self._hide_output_analysis_tooltip)
        self.output_label_prefix.bind("<Enter>", self._show_output_analysis_tooltip)
        self.output_label_prefix.bind("<Leave>", self._hide_output_analysis_tooltip)
        self.output_label_value.bind("<Enter>", self._show_output_analysis_tooltip)
        self.output_label_value.bind("<Leave>", self._hide_output_analysis_tooltip)

        # --- Dynamic Buttons ---
        self.inc_button = ttk.Button(self.main_frame, text="+", command=lambda: self._adjust_value(1), style="Small.TButton")
        self.dec_button = ttk.Button(self.main_frame, text="-", command=lambda: self._adjust_value(-1), style="Small.TButton")
        self.info_button = ttk.Button(self.main_frame, text="?", command=self._trigger_field_tooltip, style="Info.TButton")
        
        self.master.bind("<Button-1>", self._on_window_click)
        self._create_quantization_plot()
        
        self.master.after(100, self._schedule_plot_update)

    def _open_about_dialog(self):
        about_window = tk.Toplevel(self.master)
        about_window.title("About Analog to Digital Quantizer")
        about_window.transient(self.master)
        about_window.grab_set()
        about_window.resizable(False, False)
        about_window.configure(bg=self.bg_color)
        self.master.update_idletasks()
        main_x, main_y, main_w, main_h = self.master.winfo_x(), self.master.winfo_y(), self.master.winfo_width(), self.master.winfo_height()
        about_w, about_h = 450, 480
        x = main_x + (main_w // 2) - (about_w // 2)
        y = main_y + (main_h // 2) - (about_h // 2)
        about_window.geometry(f"{about_w}x{about_h}+{x}+{y}")
        tk.Label(about_window, text="About this Quantizer", font=("Arial", 12, "bold"), fg=self.text_color, bg=self.bg_color).pack(pady=10)
        tk.Label(about_window, text="Version: 1.5.0 (Stable)", font=("Arial", 11), fg=self.text_color, bg=self.bg_color).pack()
        tk.Label(about_window, text="Developed by Tiso.", font=("Arial", 11), fg=self.text_color, bg=self.bg_color).pack()
        tk.Label(about_window, text="This tool visualizes the core concepts of ADC.", font=self.italic_font, fg=self.text_color, bg=self.bg_color).pack(pady=(5, 5))
        tk.Label(about_window, text="\nHow to use:", font=("Arial", 11, "bold"), fg=self.text_color, bg=self.bg_color).pack(pady=(10, 0))
        instructions = ("1. Adjust 'Max/Min Range' and 'Bit Rate'.\n"
                        "2. Observe the real-time plot showing how quantization levels change.\n"
                        "3. Enter an 'Analog Voltage' to see its specific quantized output.\n"
                        "4. Adjust 'Sample Rate' to see how measurements over time affect the signal.\n"
                        "5. Click 'Quantize' to get the final numerical result for the current voltage.\n"
                        "6. Hover over the output for a detailed analysis of the settings.\n"
                        "7. Save the graph using the floppy disk icon in the plot toolbar.")
        tk.Label(about_window, text=instructions, wraplength=400, justify=tk.LEFT, font=("Arial", 11), fg=self.text_color, bg=self.bg_color).pack(padx=15, pady=5)
        ttk.Button(about_window, text="Close", command=about_window.destroy, style="TButton").pack(pady=(15, 10))
        self.master.wait_window(about_window)

    def _create_quantization_plot(self):
        """Creates the plot and canvas but does not perform heavy calculations."""
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.fig.patch.set_facecolor(self.plot_bg_color)
        self.ax.set_facecolor(self.plot_bg_color)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.ax.set_title("Quantization Visualization", color=self.text_color)
        self.ax.set_ylabel("Voltage (V)", color=self.text_color)
        self.ax.tick_params(axis='x', colors=self.text_color)
        self.ax.tick_params(axis='y', colors=self.text_color)
        for spine in self.ax.spines.values():
            spine.set_color(self.text_color)
        self.ax.set_ylim(-11, 11)
        self.ax.set_xlim(0, 1)
        self.fig.tight_layout()
        self.canvas.draw()

    def _schedule_plot_update(self, *args):
        if self._plot_update_job:
            self.master.after_cancel(self._plot_update_job)
        self._plot_update_job = self.master.after(50, self._execute_plot_update)

    def _execute_plot_update(self):
        if not self.fig or not self.ax: return
        self.ax.clear()
        
        # Re-apply styling
        self.ax.set_facecolor(self.plot_bg_color)
        self.ax.set_title("Quantization Visualization", color=self.text_color)
        self.ax.set_ylabel("Voltage (V)", color=self.text_color)
        self.ax.tick_params(axis='x', colors=self.text_color)
        self.ax.tick_params(axis='y', colors=self.text_color)
        for spine in self.ax.spines.values(): spine.set_color(self.text_color)

        try:
            max_r = self.field_vars["Maximum Range"].get()
            min_r = self.field_vars["Minimum Range"].get()
            bit_r = self.field_vars["Bit Rate"].get()
            analog_v = self.field_vars["Analog Voltage"].get()
            sample_rate = self.field_vars["Sample Rate"].get()
            
            if max_r <= min_r:
                self.ax.text(0.5, 0.5, "Error: Max Range must be > Min Range", ha='center', color='red', transform=self.ax.transAxes)
            else:
                self.slider_fields["Analog Voltage"].config(from_=min_r, to=max_r)
                
                duration = 1.0
                num_samples = int(sample_rate * duration)
                if num_samples < 2: num_samples = 2
                time = np.linspace(0, duration, num_samples, endpoint=False)
                
                amplitude = (max_r - min_r) / 2
                offset = (max_r + min_r) / 2
                analog_signal = amplitude * np.sin(2 * np.pi * SIGNAL_FREQUENCY * time) + offset
                self.ax.plot(time, analog_signal, color=self.analog_line_color, label='Analog Signal')
                
                num_levels = 2**bit_r
                quantization_levels = np.linspace(min_r, max_r, num_levels) if num_levels > 1 else [min_r]
                
                MAX_LINES_TO_DRAW = 100 
                if num_levels > MAX_LINES_TO_DRAW:
                    step = len(quantization_levels) // MAX_LINES_TO_DRAW
                    levels_to_draw = quantization_levels[::step]
                else:
                    levels_to_draw = quantization_levels
                
                for level in levels_to_draw:
                    self.ax.axhline(level, color=self.level_line_color, linewidth=0.7, alpha=0.5)
                
                quantized_signal = vectorized_compute_quantized_value(max_r, min_r, bit_r, analog_signal)
                self.ax.step(time, quantized_signal, where='post', color=self.quantized_line_color, label='Quantized Signal')
                
                quantized_point = compute_quantized_value(max_r, min_r, bit_r, analog_v)
                self.ax.plot(time[0], analog_v, 'x', color='#FFFF00', markersize=10, mew=2, label='Current Input')
                self.ax.plot(time[0], quantized_point, 'o', color=self.quantized_graph_output_color, markersize=10, label='Quantized Output')
                
                # FIX: Display the quantized output text directly on the graph.
                # The entire text is colored purple for emphasis to avoid parsing errors.
                text_display = f"Current Quantized Output: {quantized_point:.4f} V"
                self.ax.text(1.02, 0.8, text_display,
                             transform=self.ax.transAxes,
                             fontsize=10,
                             color=self.quantized_graph_output_color,
                             verticalalignment='top')
                
                self.ax.set_ylim(min_r - 0.1 * abs(max_r - min_r), max_r + 0.1 * abs(max_r - min_r))
                self.ax.set_xlim(0, duration)

            self.ax.set_xlabel(f"Time (s) - Sample Rate: {sample_rate} Hz", color=self.text_color)
        except (ValueError, TclError):
            self.ax.text(0.5, 0.5, "Waiting for valid inputs...", ha='center', color='yellow', transform=self.ax.transAxes)
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Plot Error: {e}", ha='center', color='red', transform=self.ax.transAxes)

        legend = self.ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
        for text in legend.get_texts(): text.set_color(self.text_color)
        self.fig.tight_layout(rect=[0, 0, 0.8, 1])
        self.canvas.draw()

    def _trigger_field_tooltip(self):
        if self.current_focused_entry:
            name = next((n for n, w in self.entry_fields.items() if w == self.current_focused_entry), None)
            if name: self._show_field_tooltip(name, self.current_focused_entry)

    def _show_field_tooltip(self, field_name, widget):
        if self.field_tooltip_window: self._hide_field_tooltip()
        message = self.field_explanations.get(field_name)
        if not message: return
        
        self.master.update_idletasks()
        x = self.info_button.winfo_rootx() + self.info_button.winfo_width() + 10
        y = self.info_button.winfo_rooty()
        
        self.field_tooltip_window = tk.Toplevel(self.master)
        self.field_tooltip_window.wm_overrideredirect(True)
        self.field_tooltip_window.wm_geometry(f"+{x}+{y}")
        frame = tk.Frame(self.field_tooltip_window, bg=self.field_tooltip_bg, relief="solid", borderwidth=1)
        frame.pack()
        tk.Label(frame, text=message, bg=self.field_tooltip_bg, fg=self.field_tooltip_fg, font=self.field_tooltip_font, justify=tk.LEFT, padx=7, pady=5, wraplength=280).pack(fill=tk.BOTH, expand=True)
        ttk.Button(frame, text="Close", command=self._hide_field_tooltip, style="TButton").pack(pady=5)

    def _hide_field_tooltip(self):
        if self.field_tooltip_window:
            self.field_tooltip_window.destroy()
            self.field_tooltip_window = None

    def _show_output_analysis_tooltip(self, event):
        if self.tooltip_window or self.output_value_text.get().strip() in ["N/A", "Error"] or not self.last_calculated_params:
            return

        p = self.last_calculated_params
        tooltip_message = ""
        try:
            voltage_range = p['max_range'] - p['min_range']
            num_levels = 2**p['bit_rate']

            if num_levels > 1 and voltage_range > 0:
                step_size = voltage_range / (num_levels - 1)
                tooltip_message += f"**Quantization Feedback:**\n- Bit Rate: {p['bit_rate']} bits ({num_levels:,} levels).\n- Voltage Step Size: {step_size:.4f} V.\n"
                tooltip_message += f"- Sample Rate: {p['sample_rate']:,} Hz.\n"
                
                tooltip_message += "\n**Theoretical Sound Quality:**\n"
                if p['bit_rate'] <= 7:
                    tooltip_message += "- Very low bit rates can lead to severe 'quantization noise' and a highly 'steppy' or 'grainy' sound, resulting in a 'crude' or 'lo-fi' audio feel."
                elif 8 <= p['bit_rate'] <= 11:
                    tooltip_message += "- Quantization noise is still perceptible, especially on quiet sounds. Audio may lack smoothness and sound somewhat 'harsh'. Often used for speech or older multimedia."
                elif 12 <= p['bit_rate'] <= 15:
                    tooltip_message += "- Good balance. Noise is significantly reduced and often imperceptible to the average listener. Sound is clear, typical for good quality streaming audio."
                else: # 16+ bits
                    tooltip_message += "- Excellent. Noise is extremely low, generally below the threshold of human hearing. Results in a smooth, high-fidelity representation comparable to professional audio (CD quality)."

                if not (p['min_range'] <= p['analog_voltage'] <= p['max_range']):
                    tooltip_message += "\n\n*Warning: Input Clipping*\n- The input voltage was outside the specified range. This would cause severe audible distortion (flat-topping of the signal)."
            else:
                tooltip_message += "**Quantization Feedback:**\n- With an invalid range or bit rate, the system cannot represent a varying signal, leading to severe loss of audio information."
        except Exception as e:
            tooltip_message = f"Error generating feedback: {e}"

        self.tooltip_window = tk.Toplevel(self.master)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{event.x_root + 15}+{event.y_root + 15}")
        tk.Label(self.tooltip_window, text=tooltip_message.strip(), bg=self.tooltip_bg, fg=self.tooltip_fg, relief="solid", borderwidth=1, font=self.tooltip_font, justify=tk.LEFT, padx=10, pady=7, wraplength=400).pack()

    def _hide_output_analysis_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def _on_entry_focus_in(self, event):
        self._hide_field_tooltip()
        self.current_focused_entry = event.widget
        row = self.current_focused_entry.grid_info()['row']
        self.inc_button.grid(row=row, column=3, padx=2, sticky="w")
        self.dec_button.grid(row=row, column=4, padx=2, sticky="w")
        self.info_button.grid(row=row, column=5, padx=2, sticky="w")

    def _on_window_click(self, event):
        if not isinstance(event.widget, (tk.Entry, ttk.Button, ttk.Scale)):
             self._hide_adjustment_buttons()
             self.master.focus_set()

    def _hide_adjustment_buttons(self):
        if self.inc_button.winfo_ismapped(): self.inc_button.grid_forget()
        if self.dec_button.winfo_ismapped(): self.dec_button.grid_forget()
        if self.info_button.winfo_ismapped(): self.info_button.grid_forget()
        self.current_focused_entry = None

    def _adjust_value(self, delta):
        if self.current_focused_entry:
            name = next((n for n, w in self.entry_fields.items() if w == self.current_focused_entry), None)
            if not name: return
            try:
                var = self.field_vars[name]
                current = float(var.get())
                if name == "Bit Rate": new_val = max(1, min(16, int(current + delta)))
                elif name == "Sample Rate": new_val = max(1000, min(100000, int(current + delta * 1000)))
                else: new_val = current + delta
                var.set(new_val)
            except ValueError: messagebox.showerror("Input Error", "Please enter a valid number.")

    def _quantize(self):
        self._hide_output_analysis_tooltip(None)
        self._hide_field_tooltip()
        try:
            params = {name: var.get() for name, var in self.field_vars.items()}
            max_r, min_r, bit_r = params["Maximum Range"], params["Minimum Range"], params["Bit Rate"]
            analog_v, sample_rate = params["Analog Voltage"], params["Sample Rate"]
            if max_r <= min_r: raise ValueError("Maximum Range must be greater than Minimum Range.")
            
            self.last_calculated_params = {
                "max_range": max_r, "min_range": min_r, "bit_rate": bit_r, 
                "analog_voltage": analog_v, "sample_rate": sample_rate
            }
            quantized_val = compute_quantized_value(max_r, min_r, bit_r, analog_v)
            self.output_value_text.set(f"{quantized_val:.4f} V")
            self.output_label_value.config(fg=self.quantized_value_color)
            if not (min_r <= analog_v <= max_r):
                messagebox.showwarning("Input Warning", f"Analog voltage ({analog_v}V) is outside the specified range [{min_r}V, {max_r}V]. Result is based on clamped value.")
            self._execute_plot_update()
        except Exception as e:
            messagebox.showerror("Calculation Error", str(e))
            self.output_value_text.set("Error")
            self.output_label_value.config(fg=self.na_color)

    def _clear_inputs(self):
        defaults = {"Maximum Range": 10.0, "Minimum Range": -10.0, "Bit Rate": 8, "Analog Voltage": 5.0, "Sample Rate": 44100}
        for field, value in defaults.items(): self.field_vars[field].set(value)
        self.output_value_text.set("N/A")
        self.output_label_value.config(fg=self.na_color)
        self.last_calculated_params = None
        self._schedule_plot_update()

    def _on_window_configure(self, event):
        if event.widget == self.master:
            # 'zoomed' is the state for a maximized window on Windows
            state = self.master.state()
            if state == 'zoomed':
                self.about_button.grid()
            else:
                self.about_button.grid_forget()

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantizerApp(root)
    root.mainloop()