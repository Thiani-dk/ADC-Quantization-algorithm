import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter.font as tkFont

# --- Algorithm for Quantization ---
def compute_quantized_value(max_range, min_range, bit_rate, analog_voltage):
    """
    Computes the quantized value of a given analog voltage.

    Args:
        max_range (float): The maximum voltage of the analog input range (V).
        min_range (float): The minimum voltage of the analog input range (V).
        bit_rate (int): The number of bits for quantization.
        analog_voltage (float): The input analog voltage to be quantized (V).

    Returns:
        float: The quantized voltage value.
    Raises:
        ValueError: If input parameters are invalid.
    """
    if max_range <= min_range:
        # For computation, if ranges are invalid, map to min_range or handle as error
        # A proper error should ideally be caught before this call for a meaningful calculation
        # For plotting, it's handled in _update_quantization_plot
        return min_range 
    if bit_rate <= 0:
        raise ValueError("Bit Rate must be a positive integer.")

    num_quantization_levels = 2**bit_rate
    
    if num_quantization_levels <= 1: # If only one level, all values map to min_range
        return min_range
    
    step_size = (max_range - min_range) / (num_quantization_levels - 1)

    normalized_voltage = (analog_voltage - min_range) / (max_range - min_range)
    quantized_level_index = round(normalized_voltage * (num_quantization_levels - 1))

    # Ensure the index is within valid bounds (0 to num_quantization_levels - 1)
    quantized_level_index = max(0, min(int(quantized_level_index), num_quantization_levels - 1))

    quantized_voltage = min_range + (quantized_level_index * step_size)

    return quantized_voltage


# --- GUI Application ---
class QuantizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Analog to Digital Quantizer")

        # --- Define Colors and Fonts ---
        self.bg_color = "#2b2b2b"  # Dark background
        self.frame_bg_color = "#3c3f41" # Slightly lighter dark for the main frame
        self.text_color = "#cccccc" # Light gray text
        self.entry_bg_color = "#454545" # Darker gray for entry fields
        self.entry_fg_color = "#ffffff" # White text in entry fields
        self.button_bg_color = "#6a90c9" # Default button color (will be overridden for specific buttons)
        self.button_fg_color = "#ffffff" # White text on buttons
        self.button_active_bg_color = "#7da7e3" # Lighter blue on hover

        self.quantize_button_color = "#4CAF50"  # Green for Quantize
        self.quantize_button_hover_color = "#66BB6A"
        self.clear_button_color = "#F44336"     # Red for Clear Selection
        self.clear_button_hover_color = "#E57373"

        # Tooltip colors
        self.tooltip_bg = "#FFFFCC" # Light yellow for output analysis tooltip
        self.tooltip_fg = "#333333" # Dark grey text
        self.field_tooltip_bg = "#E0FFFF" # Light cyan for field explanations
        self.field_tooltip_fg = "#333333"
        
        # Info button colors
        self.info_button_fg_color = "#FFFF00" # Yellow for the question mark
        self.info_button_active_fg_color = "#FFA500" # Orange for hover

        # Quantized output specific color (GUI text and plot marker/label)
        # This color is for the GUI output label when a value is displayed
        self.quantized_value_color = "#FFFF00" # Changed to Yellow as requested
        
        # This color is for the 'N/A' output text (now same as quantized_value_color)
        self.na_color = "#FFFF00" # Pure Yellow

        # Color specifically for the quantized output point on the graph
        self.quantized_graph_output_color = "#9400D3" # DarkViolet (Purple)

        # Fonts - Underlined title font
        self.title_font = tkFont.Font(family="Arial", size=18, weight="bold", underline=True)
        self.label_font = ("Arial", 11)
        self.entry_font = ("Consolas", 11)
        self.button_font = ("Arial", 10, "bold")
        self.output_label_font = ("Arial", 13, "bold")
        self.output_value_font = ("Consolas", 14, "bold")
        self.tooltip_font = ("Arial", 9)
        self.field_tooltip_font = ("Arial", 9)
        self.close_button_font = ("Arial", 8) # Smaller font for close button
        self.italic_font = tkFont.Font(family="Arial", size=11, slant="italic") # Increased font for italic text

        # Plot specific colors
        self.plot_bg_color = "#1e1e1e" # Darker background for plot
        self.analog_line_color = "#32CD32" # Changed to Lime Green as requested
        self.quantized_line_color = "#FF0000" # Changed to Bright Red as requested
        self.level_line_color = "#DDDDDD" # Very Light Gray for quantization levels (remains)

        # Configure the root window background
        self.master.configure(bg=self.bg_color)
        
        # Set a minimum size for the root window to prevent extreme squeezing
        self.master.minsize(800, 500) 

        # Configure root window for resizability (main_frame on left, plot_frame on right)
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1) # Main frame column
        self.master.grid_columnconfigure(1, weight=1) # Plot frame column
        
        # Bind the <Configure> event to check window state for about button
        self.master.bind('<Configure>', self._on_window_configure)

        # Dictionary to hold Entry widgets and their associated Tkinter variables
        self.entry_fields = {}
        self.field_vars = {} # To hold DoubleVar/IntVar for linking entries and sliders
        self.slider_fields = {} # To hold slider widgets
        self.current_focused_entry = None
        self.tooltip_window = None
        self.field_tooltip_window = None
        self.last_calculated_params = None

        # Matplotlib plot variables
        self.fig = None
        self.ax = None
        self.canvas = None
        self.toolbar = None
        self._quantized_output_text_label = None # To hold the plot text object

        # --- Explanations for each input field ---
        self.field_explanations = {
            "Maximum Range": "This defines the highest voltage value that your analog-to-digital converter (ADC) can measure. Setting it too low for a given signal can lead to clipping (distortion) if the signal exceeds this range.",
            "Minimum Range": "This defines the lowest voltage value that your analog-to-digital converter (ADC) can measure. Setting it too high for a given signal can also lead to clipping or a loss of lower amplitude information if the signal falls below this range.",
            "Bit Rate": "The 'Bit Rate' (or bit depth) determines the number of quantization levels available. More bits mean more levels, leading to finer resolution and less quantization error (noise), which improves the quality of the digitized signal. For example, 8 bits give 256 levels, while 16 bits give 65,536 levels. Lower bit rates can result in a 'grainy' or 'noisy' sound due to larger quantization steps.",
            "Analog Voltage": "This is the instantaneous voltage of the analog signal that you want to convert into a digital value. The ADC will approximate this voltage to the nearest available quantization level within the defined range."
        }

        # --- Main Frame for layout and border ---
        self.main_frame = tk.Frame(master, padx=20, pady=20, bd=2, relief="groove", bg=self.frame_bg_color, highlightbackground=self.text_color, highlightthickness=1)
        self.main_frame.grid(row=0, column=0, sticky="nsew", pady=20, padx=20) # Use grid for main window layout

        # --- Plot Frame ---
        self.plot_frame = tk.Frame(master, bg=self.plot_bg_color)
        self.plot_frame.grid(row=0, column=1, sticky="nsew", pady=20, padx=(0, 20)) # Use grid for main window layout

        # Configure columns within main_frame for responsiveness
        self.main_frame.grid_columnconfigure(0, weight=0) # Label column - fixed
        self.main_frame.grid_columnconfigure(1, weight=1) # Entry column - expands
        self.main_frame.grid_columnconfigure(2, weight=0) # Unit column - fixed
        self.main_frame.grid_columnconfigure(3, weight=0) # Button +/-
        self.main_frame.grid_columnconfigure(4, weight=0) # Button +/-
        self.main_frame.grid_columnconfigure(5, weight=0) # Button ? / About (flexible)
        self.main_frame.grid_columnconfigure(6, weight=2) # Slider column - expands more

        # --- Title Label ---
        # Increased pady for title to give more space
        self.title_label = tk.Label(self.main_frame, text="Analog to Digital Quantizer", font=self.title_font, fg=self.text_color, bg=self.frame_bg_color)
        self.title_label.grid(row=0, column=0, columnspan=7, pady=(15, 20), sticky="ew") # Spans all 7 columns

        # --- About Button (Initially hidden) ---
        self.about_button = ttk.Button(self.main_frame, text="About", command=self._open_about_dialog, style="TButton")
        # Don't grid it initially; visibility will be controlled by _on_window_configure
        self.about_button.grid(row=0, column=6, padx=5, pady=5, sticky="ne") # Placed top right
        self.about_button.grid_forget() # Hide it at startup

        # --- Input Fields, Labels, and Sliders ---
        input_specs = [
            ("Maximum Range", "V", 10.0),
            ("Minimum Range", "V", -10.0),
            ("Bit Rate", "Bit", 8),
            ("Analog Voltage", "V", 5.0)
        ]

        row_num = 1
        for text, unit, default_value in input_specs:
            self.main_frame.grid_rowconfigure(row_num, weight=1) # Make input rows expand vertically
            tk.Label(self.main_frame, text=f"{text}:", font=self.label_font, fg=self.text_color, bg=self.frame_bg_color).grid(row=row_num, column=0, sticky="w", pady=5)

            # Create Tkinter variable for the field
            if text == "Bit Rate":
                var = tk.IntVar(value=default_value)
                var.trace_add('write', lambda name, index, mode, var=var: self._update_quantization_plot_from_var(var))
            else:
                var = tk.DoubleVar(value=default_value)
                var.trace_add('write', lambda name, index, mode, var=var: self._update_quantization_plot_from_var(var))
            self.field_vars[text] = var

            entry = tk.Entry(self.main_frame, width=15, bd=1, relief="solid",
                             bg=self.entry_bg_color, fg=self.entry_fg_color,
                             insertbackground=self.entry_fg_color,
                             font=self.entry_font,
                             textvariable=var) # Link Entry to variable
            entry.grid(row=row_num, column=1, pady=5, padx=5, sticky="ew") # Allow entry to expand

            tk.Label(self.main_frame, text=unit, font=self.label_font, fg=self.text_color, bg=self.frame_bg_color).grid(row=row_num, column=2, sticky="w")

            self.entry_fields[text] = entry
            entry.bind("<FocusIn>", self._on_entry_focus_in)

            # Add slider for relevant fields
            if text in ["Maximum Range", "Minimum Range", "Analog Voltage"]:
                slider_min = -20.0 
                slider_max = 20.0 

                slider = ttk.Scale(self.main_frame,
                                   from_=slider_min, to=slider_max,
                                   orient="horizontal",
                                   variable=var, # Link slider to variable
                                   command=lambda val, var=var: self._update_quantization_plot_from_var(var))
                slider.grid(row=row_num, column=6, padx=5, pady=5, sticky="ew")
                self.slider_fields[text] = slider
                var.set(default_value) # Set initial value after binding everything
            else:
                var.set(default_value)

            row_num += 1

        # --- Configure Button Styles for ttk ---
        style = ttk.Style()
        style.theme_use('clam')

        style.configure("TButton",
                        background=self.button_bg_color,
                        foreground=self.button_fg_color,
                        font=self.button_font,
                        padding=5,
                        relief="flat")
        style.map("TButton",
                  background=[('active', self.button_active_bg_color)],
                  foreground=[('active', self.button_fg_color)])

        style.configure("Quantize.TButton",
                        background=self.quantize_button_color,
                        foreground=self.button_fg_color)
        style.map("Quantize.TButton",
                  background=[('active', self.quantize_button_hover_color)],
                  foreground=[('active', self.button_fg_color)])

        style.configure("Clear.TButton",
                        background=self.clear_button_color,
                        foreground=self.button_fg_color)
        style.map("Clear.TButton",
                  background=[('active', self.clear_button_hover_color)],
                  foreground=[('active', self.button_fg_color)])

        style.configure("Small.TButton",
                        font=("Arial", 9, "bold"),
                        width=4,
                        padding=2)
        style.map("Small.TButton",
                  background=[('active', self.button_active_bg_color)],
                  foreground=[('active', self.button_fg_color)])
        
        style.configure("Info.TButton",
                        background=self.frame_bg_color,
                        foreground=self.info_button_fg_color,
                        font=("Arial", 12, "bold"),
                        width=3,
                        padding=0,
                        relief="flat")
        style.map("Info.TButton",
                  background=[('active', self.frame_bg_color)],
                  foreground=[('active', self.info_button_active_fg_color)])
        
        # --- Quantize and Clear Buttons (re-positioned and centered) ---
        # Create a sub-frame to group and center these buttons
        self.action_buttons_frame = tk.Frame(self.main_frame, bg=self.frame_bg_color)
        # Position this frame below the input fields, spanning across columns for centering
        self.action_buttons_frame.grid(row=row_num, column=0, columnspan=7, pady=(15, 5), sticky="ew")
        
        # Configure the columns within this sub-frame to center the buttons
        self.action_buttons_frame.grid_columnconfigure(0, weight=1) # Left spacer
        self.action_buttons_frame.grid_columnconfigure(1, weight=0) # Quantize Button
        self.action_buttons_frame.grid_columnconfigure(2, weight=0) # Clear Button
        self.action_buttons_frame.grid_columnconfigure(3, weight=1) # Right spacer

        self.quantize_button = ttk.Button(self.action_buttons_frame, text="Quantize", command=self._quantize, width=15, style="Quantize.TButton")
        self.quantize_button.grid(row=0, column=1, padx=5, pady=5) # Use row 0 within its own frame

        self.clear_button = ttk.Button(self.action_buttons_frame, text="Clear", command=self._clear_inputs, width=15, style="Clear.TButton")
        self.clear_button.grid(row=0, column=2, padx=5, pady=5) # Use row 0 within its own frame

        row_num += 1 # Increment row_num for elements below action buttons frame


        # --- Output Section ---
        self.output_frame = tk.Frame(self.main_frame, bg=self.frame_bg_color)
        self.output_frame.grid(row=row_num + 1, column=0, columnspan=7, pady=20, sticky="ew") # Spans all 7 columns

        # Label for the static "Quantized Voltage:" text
        self.output_label_prefix = tk.Label(self.output_frame, text="Quantized Voltage: ",
                                            font=self.output_label_font,
                                            fg=self.text_color, bg=self.frame_bg_color)
        self.output_label_prefix.pack(side=tk.LEFT, padx=(0,0))

        # Label for the dynamic value and unit (will now always be yellow)
        self.output_value_text = tk.StringVar()
        self.output_value_text.set("N/A")
        self.output_label_value = tk.Label(self.output_frame, textvariable=self.output_value_text,
                                           font=self.output_value_font,
                                           fg=self.na_color, # Start with yellow for N/A
                                           bg=self.frame_bg_color)
        self.output_label_value.pack(side=tk.LEFT)

        # --- Bind hover events to the output frame for the tooltip ---
        self.output_frame.bind("<Enter>", self._show_output_analysis_tooltip)
        self.output_frame.bind("<Leave>", self._hide_output_analysis_tooltip)
        self.output_label_prefix.bind("<Enter>", self._show_output_analysis_tooltip)
        self.output_label_prefix.bind("<Leave>", self._hide_output_analysis_tooltip)
        self.output_label_value.bind("<Enter>", self._show_output_analysis_tooltip)
        self.output_label_value.bind("<Leave>", self._hide_output_analysis_tooltip)


        # --- Increment/Decrement and Info Buttons (dynamic) ---
        self.inc_button = ttk.Button(self.main_frame, text="+", command=lambda: self._adjust_value(1), style="Small.TButton")
        self.dec_button = ttk.Button(self.main_frame, text="-", command=lambda: self._adjust_value(-1), style="Small.TButton")
        self.info_button = ttk.Button(self.main_frame, text="?", command=self._trigger_field_tooltip, style="Info.TButton")

        self.inc_button.grid_forget()
        self.dec_button.grid_forget()
        self.info_button.grid_forget()

        self.master.bind("<Button-1>", self._on_window_click)

        # Initialize the quantization plot
        self._create_quantization_plot()
        self._update_quantization_plot() # Initial plot draw with default values


    # --- New: About Dialog ---
    def _open_about_dialog(self):
        """Opens a Toplevel window with information about the application."""
        about_window = tk.Toplevel(self.master)
        about_window.title("About Analog to Digital Quantizer") # This title is for the OS window frame
        about_window.transient(self.master) # Make it appear on top of the main window
        about_window.grab_set() # Prevent interaction with main window until closed
        about_window.resizable(False, False) # Don't allow resizing
        about_window.configure(bg=self.bg_color)

        # Center the about window relative to the main window
        # Increased width and height to fit more info
        about_window_width = 450
        about_window_height = 450 
        self.master.update_idletasks() # Ensure sizes are calculated
        x = self.master.winfo_x() + (self.master.winfo_width() // 2) - (about_window_width // 2)
        y = self.master.winfo_y() + (self.master.winfo_height() // 2) - (about_window_height // 2)
        about_window.geometry(f"{about_window_width}x{about_window_height}+{x}+{y}")


        # Changed title inside the white section as requested
        tk.Label(about_window, text="About this Quantizer", font=("Arial", 12, "bold"), fg=self.text_color, bg=self.bg_color).pack(pady=10)
        # Increased font size for version and developer info
        tk.Label(about_window, text="Version: 1.2.0", font=("Arial", 11), fg=self.text_color, bg=self.bg_color).pack()
        tk.Label(about_window, text="Developed by Tiso.", font=("Arial", 11), fg=self.text_color, bg=self.bg_color).pack() # Updated author
        
        # New italicized, centered text (font already increased in __init__)
        tk.Label(about_window, text="its simply just like any calculator", font=self.italic_font, fg=self.text_color, bg=self.bg_color).pack(pady=(5, 5))

        # Increased font size for "How to use:" title
        tk.Label(about_window, text="\nHow to use:", font=("Arial", 11, "bold"), fg=self.text_color, bg=self.bg_color).pack(pady=(10, 0))
        instructions = (
            "1. Adjust 'Max/Min Range' and 'Bit Rate' using the entry boxes, sliders, or +/- buttons.\n"
            "2. Observe the real-time plot showing how quantization levels change.\n"
            "3. Enter an 'Analog Voltage' or use its slider to see its quantized output.\n"
            "4. Click 'Quantize' to get the final numerical result.\n"
            "5. Hover over the output for detailed analysis.\n"
            "6. You can save the graph as an image using the save icon (floppy disk) in the plot toolbar."
        )
        # Adjusted wraplength to fit the new width and increased font size for instructions
        tk.Label(about_window, text=instructions, wraplength=400, justify=tk.LEFT, font=("Arial", 11), fg=self.text_color, bg=self.bg_color).pack(padx=15, pady=5)
        
        # Close button at the bottom, centered with adequate padding
        ttk.Button(about_window, text="Close", command=about_window.destroy, style="TButton").pack(pady=(15, 10))

        # Focus the dialog and wait until it's closed
        self.master.wait_window(about_window)


    # --- Matplotlib Plotting Methods ---
    def _create_quantization_plot(self):
        """Initializes the Matplotlib plot and embeds it in the Tkinter frame."""
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.fig.patch.set_facecolor(self.plot_bg_color)
        self.ax.set_facecolor(self.plot_bg_color)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X, anchor="s")

        self.ax.set_title("Quantization Visualization", color=self.text_color)
        self.ax.set_xlabel("Time", color=self.text_color)
        self.ax.set_ylabel("Voltage (V)", color=self.text_color)
        
        self.ax.tick_params(axis='x', colors=self.text_color)
        self.ax.tick_params(axis='y', colors=self.text_color)
        self.ax.spines['bottom'].set_color(self.text_color)
        self.ax.spines['left'].set_color(self.text_color)
        self.ax.spines['top'].set_color(self.text_color)
        self.ax.spines['right'].set_color(self.text_color)

        self.ax.plot(np.linspace(0, 1, 100), np.sin(np.linspace(0, 2 * np.pi, 100)), color=self.analog_line_color, label='Analog Signal')
        self.ax.set_ylim(-1.5, 1.5)
        
        # --- Legend placement change ---
        # This places the legend outside the plot area, to the right.
        # bbox_to_anchor=(1.02, 1) means just outside the top-right corner.
        # borderaxespad=0 means no padding between legend and axes.
        self.ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, labelcolor=self.text_color)
        
        # Adjust subplot parameters to make room for the legend and new text
        self.fig.subplots_adjust(right=0.8) # Increased right margin

        self.fig.tight_layout()


    def _update_quantization_plot_from_var(self, var):
        """
        Callback for Tkinter variables. Triggers plot update.
        Needed because trace_add doesn't pass a fixed event object.
        """
        self._update_quantization_plot()

    def _update_quantization_plot(self):
        """Updates the quantization visualization based on current input values."""
        if not self.fig or not self.ax:
            return

        # Clear existing text label if it exists
        if hasattr(self, '_quantized_output_text_label') and self._quantized_output_text_label:
            self._quantized_output_text_label.remove()
            self._quantized_output_text_label = None # Clear reference

        self.ax.clear()

        # Reset plot aesthetics after clearing
        self.ax.set_facecolor(self.plot_bg_color)
        self.ax.set_title("Quantization Visualization", color=self.text_color)
        self.ax.set_xlabel("Time", color=self.text_color)
        self.ax.set_ylabel("Voltage (V)", color=self.text_color)
        self.ax.tick_params(axis='x', colors=self.text_color)
        self.ax.tick_params(axis='y', colors=self.text_color)
        self.ax.spines['bottom'].set_color(self.text_color)
        self.ax.spines['left'].set_color(self.text_color)
        self.ax.spines['top'].set_color(self.text_color)
        self.ax.spines['right'].set_color(self.text_color)
        
        try:
            # Get values directly from Tkinter variables
            max_r = self.field_vars["Maximum Range"].get()
            min_r = self.field_vars["Minimum Range"].get()
            bit_r = self.field_vars["Bit Rate"].get()
            analog_v_single_point = self.field_vars["Analog Voltage"].get()

            # Basic validation for plot rendering
            if max_r <= min_r:
                self.ax.text(0.5, 0.5, "Error: Max Range must be > Min Range", ha='center', va='center', color='red', transform=self.ax.transAxes)
                self.ax.set_ylim(-1.5, 1.5)
                self.ax.set_xlim(0, 2*np.pi)
                # Ensure legend is still drawn even on error
                self.ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, labelcolor=self.text_color)
                self.fig.subplots_adjust(right=0.8) # Ensure right margin is consistent
                self.canvas.draw()
                return 

            bit_r = max(1, bit_r) # Minimum 1 bit for plot logic

            # Update slider ranges for Analog Voltage dynamically
            if "Analog Voltage" in self.slider_fields:
                # Ensure slider 'from_' is not greater than 'to_'
                if min_r < max_r:
                    self.slider_fields["Analog Voltage"].config(from_=min_r, to=max_r)
                else: # Fallback for invalid ranges, maybe set a default or disable
                    self.slider_fields["Analog Voltage"].config(from_=min_r, to=min_r + 0.1) # Small valid range

                # Keep the analog voltage within the new range
                current_analog_v = self.field_vars["Analog Voltage"].get()
                if not (min_r <= current_analog_v <= max_r):
                    # Snap to nearest valid point if outside new range
                    if current_analog_v < min_r: self.field_vars["Analog Voltage"].set(min_r)
                    if current_analog_v > max_r: self.field_vars["Analog Voltage"].set(max_r)


            # Generate a sine wave as the analog signal
            time = np.linspace(0, 2 * np.pi, 200)
            analog_signal = np.sin(time) * ((max_r - min_r) / 2) + (max_r + min_r) / 2
            
            self.ax.plot(time, analog_signal, color=self.analog_line_color, label='Analog Signal')

            # Calculate quantization levels
            num_levels = 2**bit_r
            if num_levels == 1:
                quantization_levels = [min_r]
            else:
                quantization_levels = np.linspace(min_r, max_r, num_levels)

            for level in quantization_levels:
                # Use the new light gray color for the level lines
                self.ax.axhline(level, color=self.level_line_color, linestyle='--', linewidth=0.7, alpha=0.6)

            # Quantize the analog signal
            quantized_signal = []
            for val in analog_signal:
                quantized_val = compute_quantized_value(max_r, min_r, bit_r, val)
                quantized_signal.append(quantized_val)

            self.ax.step(time, quantized_signal, where='mid', color=self.quantized_line_color, label='Quantized Signal')

            # Mark the specific input analog voltage if within range
            if min_r <= analog_v_single_point <= max_r:
                 quantized_single_point = compute_quantized_value(max_r, min_r, bit_r, analog_v_single_point)
                 # Mark input point with an 'X' using YELLOW
                 self.ax.plot(time[0], analog_v_single_point, 'x', color='#FFFF00', markersize=10, mew=2, label='Current Input')
                 # Mark quantized output point with an 'O' using purple
                 self.ax.plot(time[0], quantized_single_point, 'o', color=self.quantized_graph_output_color, markersize=10, label='Quantized Output') 
                 
                 # Display quantized output value below the legend
                 text_display = f"Current Quantized Output: {quantized_single_point:.2f}V"
                 self._quantized_output_text_label = self.ax.text(1.02, 0.7, text_display, # Adjusted y to 0.7 for ample space
                                                                  transform=self.ax.transAxes,
                                                                  fontsize=10, 
                                                                  color=self.text_color, 
                                                                  verticalalignment='top')
                 

            self.ax.set_ylim(min_r - 0.1 * abs(max_r - min_r), max_r + 0.1 * abs(max_r - min_r))
            self.ax.set_xlim(0, 2 * np.pi) # Ensure x-axis limits are consistent

        except ValueError as e:
            self.ax.text(0.5, 0.5, f"Invalid Numerical Input: {e}", ha='center', va='center', color='red', transform=self.ax.transAxes)
            self.ax.set_ylim(-1.5, 1.5)
            self.ax.set_xlim(0, 2*np.pi)
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Plot Error: {e}", ha='center', va='center', color='red', transform=self.ax.transAxes)
            self.ax.set_ylim(-1.5, 1.5)
            self.ax.set_xlim(0, 2*np.pi)

        # --- Legend placement (repeated after clear to ensure it's always set) ---
        self.ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, labelcolor=self.text_color)
        self.fig.subplots_adjust(right=0.8) # Adjust right margin to fit legend and new text
        self.fig.tight_layout() # Call tight_layout last
        self.canvas.draw()


    # --- Rest of your existing methods (with minor adjustments for variable use) ---
    def _trigger_field_tooltip(self):
        if self.current_focused_entry:
            field_name = None
            for name, entry_widget in self.entry_fields.items():
                if entry_widget == self.current_focused_entry:
                    field_name = name
                    break
            if field_name:
                self._show_field_tooltip(field_name, self.current_focused_entry)

    def _show_field_tooltip(self, field_name, entry_widget):
        if self.field_tooltip_window:
            self._hide_field_tooltip()

        if not field_name or field_name not in self.field_explanations:
            return

        tooltip_message = self.field_explanations[field_name]

        x = entry_widget.winfo_rootx()
        y = entry_widget.winfo_rooty()
        
        # Position tooltip to the right of the info button (column 5)
        # Assuming info button is column 5, and its width is small.
        # We need to find its root x and y
        info_button_x = self.info_button.winfo_rootx()
        info_button_y = self.info_button.winfo_rooty()
        
        tooltip_x = info_button_x + self.info_button.winfo_width() + 10 # 10 pixels padding
        tooltip_y = info_button_y # Align top with button

        self.field_tooltip_window = tk.Toplevel(self.master)
        self.field_tooltip_window.wm_overrideredirect(True)
        self.field_tooltip_window.wm_geometry(f"+{tooltip_x}+{tooltip_y}")

        tooltip_frame = tk.Frame(self.field_tooltip_window,
                                 background=self.field_tooltip_bg,
                                 relief="solid", borderwidth=1)
        tooltip_frame.pack(padx=0, pady=0)

        label = tk.Label(tooltip_frame, text=tooltip_message,
                         background=self.field_tooltip_bg, foreground=self.field_tooltip_fg,
                         font=self.field_tooltip_font,
                         justify=tk.LEFT,
                         padx=7, pady=5,
                         wraplength=280)
        label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        close_button = ttk.Button(tooltip_frame, text="Close",
                                  command=self._hide_field_tooltip,
                                  style="TButton")
        close_button.pack(side=tk.BOTTOM, pady=5, padx=5)

    def _hide_field_tooltip(self):
        if self.field_tooltip_window:
            self.field_tooltip_window.destroy()
            self.field_tooltip_window = None

    def _show_output_analysis_tooltip(self, event):
        if self.tooltip_window:
            return

        current_output_str = self.output_value_text.get().strip()
        
        if not current_output_str or current_output_str in ["N/A", "Error"] or self.last_calculated_params is None:
            return

        tooltip_message = ""
        
        try:
            max_r = self.last_calculated_params["max_range"]
            min_r = self.last_calculated_params["min_range"]
            bit_r = self.last_calculated_params["bit_rate"]
            analog_v = self.last_calculated_params["analog_voltage"]

            voltage_range = max_r - min_r
            num_quantization_levels = 2**bit_r
            
            if num_quantization_levels > 1 and voltage_range > 0:
                step_size = voltage_range / (num_quantization_levels - 1)
                
                tooltip_message += "**Quantization Feedback:**\n"
                tooltip_message += f"- Bit Rate: {bit_r} bits. This provides {num_quantization_levels} discrete levels (e.g., a 16-bit system has 65,536 levels). More bits generally lead to higher fidelity and less quantization noise (often perceived as a 'hiss' or 'graininess').\n"
                tooltip_message += f"- Voltage Step Size: {step_size:.4f} V. This is the smallest change in voltage that can be represented. A smaller step size means a more accurate representation of the analog signal.\n"

                tooltip_message += "\n**Theoretical Sound Quality:**\n"
                if bit_r <= 7:
                    tooltip_message += "- With very low bit rates (e.g., 1-7 bits), quantization steps are very large. This can lead to severe 'quantization noise' and a highly 'steppy' or 'grainy' sound, particularly noticeable in quiet passages. The digital representation will have a very limited dynamic range, resulting in a 'crude' or 'lo-fi' audio feel. Imagine an extremely basic digital recording.\n"
                elif 8 <= bit_r <= 11:
                    tooltip_message += "- At this bit rate, quantization noise is still quite perceptible, especially on quiet sounds or decays. The audio may lack smoothness and sound somewhat 'harsh' or 'digital'. It's often used for speech or less critical audio where file size is paramount (e.g., telephone audio or older multimedia presentations).\n"
                elif 12 <= bit_r <= 15:
                    tooltip_message += "- This range offers a good balance. Quantization noise is significantly reduced and often imperceptible to the average listener for most audio. The sound should be clear and generally pleasant, though critical listening might reveal minor imperfections in very high-fidelity recordings. This would be typical for many streaming services or good quality compressed audio.\n"
                else:
                    tooltip_message += "- With 16 bits or more, the quantization noise is extremely low and generally below the threshold of human hearing (the 'noise floor'). This results in a very smooth, accurate, and high-fidelity digital representation. The sound should be clean, detailed, and have excellent dynamic range, comparable to professional audio recordings or CD quality (16-bit, 44.1 kHz).\n"
                    
                if not (min_r <= analog_v <= max_r):
                    tooltip_message += "\n*Warning: Input Clipping*\n"
                    tooltip_message += "- The input analog voltage was outside the specified range. This would result in 'clipping' during conversion, leading to severe audible distortion at the peaks of the signal (flat-topping or flat-bottoming). This sounds like a harsh, fuzzy distortion, especially on loud sounds.\n"

            else:
                tooltip_message += "**Quantization Feedback:**\n"
                tooltip_message += "- With very few bits or an invalid voltage range, the system cannot accurately represent a varying analog signal. The output will likely be highly distorted, clipped, or simply a constant value, leading to severe loss of audio information. The resulting 'sound' would be unusable.\n"

        except (ValueError, KeyError):
            tooltip_message += "\n(Cannot provide detailed feedback due to missing or invalid input values from the last calculation.)\n"
        except Exception as e:
            tooltip_message += f"\n(Error generating detailed feedback: {e})\n"

        tooltip_message = tooltip_message.strip()

        self.tooltip_window = tk.Toplevel(self.master)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{event.x_root + 15}+{event.y_root + 15}")

        label = tk.Label(self.tooltip_window, text=tooltip_message,
                         background=self.tooltip_bg, foreground=self.tooltip_fg,
                         relief="solid", borderwidth=1,
                         font=self.tooltip_font,
                         justify=tk.LEFT,
                         padx=10, pady=7,
                         wraplength=400)
        label.pack()

    def _hide_output_analysis_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def _on_entry_focus_in(self, event):
        self._hide_field_tooltip()
        self.current_focused_entry = event.widget
        grid_info = self.current_focused_entry.grid_info()
        row = grid_info['row']

        # Place buttons relative to the focused entry
        self.inc_button.grid(row=row, column=3, padx=2, sticky="w")
        self.dec_button.grid(row=row, column=4, padx=2, sticky="w")
        self.info_button.grid(row=row, column=5, padx=2, sticky="w")

    def _on_window_click(self, event):
        is_entry = isinstance(event.widget, tk.Entry)
        is_inc_button = (event.widget == self.inc_button.winfo_containing(event.x_root, event.y_root))
        is_dec_button = (event.widget == self.dec_button.winfo_containing(event.x_root, event.y_root))
        is_info_button = (event.widget == self.info_button.winfo_containing(event.x_root, event.y_root))

        # Check if click was on a slider
        is_slider = False
        for s in self.slider_fields.values():
            if s.winfo_containing(event.x_root, event.y_root) == s:
                is_slider = True
                break

        is_tooltip_click = False
        if self.tooltip_window:
            tooltip_x1 = self.tooltip_window.winfo_x()
            tooltip_y1 = self.tooltip_window.winfo_y()
            tooltip_x2 = tooltip_x1 + self.tooltip_window.winfo_width()
            tooltip_y2 = tooltip_y1 + self.tooltip_window.winfo_height()
            if tooltip_x1 <= event.x_root <= tooltip_x2 and tooltip_y1 <= event.y_root <= tooltip_y2:
                is_tooltip_click = True
        
        if self.field_tooltip_window:
            field_tooltip_x1 = self.field_tooltip_window.winfo_x()
            field_tooltip_y1 = self.field_tooltip_window.winfo_y()
            field_tooltip_x2 = field_tooltip_x1 + self.field_tooltip_window.winfo_width()
            field_tooltip_y2 = field_tooltip_y1 + self.field_tooltip_window.winfo_height()
            if field_tooltip_x1 <= event.x_root <= field_tooltip_x2 and field_tooltip_y1 <= event.y_root <= field_tooltip_y2:
                is_tooltip_click = True

        is_plot_click = False
        if self.canvas_widget:
            # winfo_rootx/y give absolute screen coordinates
            plot_x1 = self.canvas_widget.winfo_rootx()
            plot_y1 = self.canvas_widget.winfo_rooty()
            plot_x2 = plot_x1 + self.canvas_widget.winfo_width()
            plot_y2 = plot_y1 + self.canvas_widget.winfo_height()
            if plot_x1 <= event.x_root <= plot_x2 and plot_y1 <= event.y_root <= plot_y2:
                is_plot_click = True
        
        if self.toolbar:
            toolbar_x1 = self.toolbar.winfo_rootx()
            toolbar_y1 = self.toolbar.winfo_y() # Use winfo_y for relative to parent
            toolbar_x2 = toolbar_x1 + self.toolbar.winfo_width()
            toolbar_y2 = toolbar_y1 + self.toolbar.winfo_height()
            # Convert event.x/y to coordinates relative to the toolbar's root for precise check
            # This is complex, simply check if the click is within the toolbar's *absolute screen* bounds.
            if toolbar_x1 <= event.x_root <= toolbar_x2 and toolbar_y1 <= event.y_root <= toolbar_y2:
                is_plot_click = True
        
        is_about_button_click = (event.widget == self.about_button.winfo_containing(event.x_root, event.y_root))


        if not (is_entry or is_inc_button or is_dec_button or is_info_button or is_tooltip_click or is_plot_click or is_slider or is_about_button_click):
            self._hide_adjustment_buttons()
            self._hide_output_analysis_tooltip(None)
            self._hide_field_tooltip()
            self.master.focus_set()


    def _hide_adjustment_buttons(self):
        if self.inc_button.winfo_ismapped():
            self.inc_button.grid_forget()
        if self.dec_button.winfo_ismapped():
            self.dec_button.grid_forget()
        if self.info_button.winfo_ismapped():
            self.info_button.grid_forget()
        self.current_focused_entry = None


    def _adjust_value(self, delta):
        if self.current_focused_entry:
            field_name = None
            for name, entry_widget in self.entry_fields.items():
                if entry_widget == self.current_focused_entry:
                    field_name = name
                    break
            
            if field_name:
                try:
                    current_var = self.field_vars[field_name]
                    current_value = current_var.get()
                    
                    if field_name == "Bit Rate":
                        new_value = int(current_value) + delta
                        if new_value < 1:
                            new_value = 1
                        current_var.set(new_value)
                    else:
                        new_value = float(current_value) + delta
                        current_var.set(f"{new_value:.1f}") # Ensure float format for entry
                    
                except ValueError:
                    messagebox.showerror("Input Error", "Please enter a valid number in the selected field before adjusting.")
            else:
                messagebox.showinfo("Selection", "Please select an input field to adjust.")


    def _quantize(self):
        self._hide_output_analysis_tooltip(None)
        self._hide_field_tooltip()
        try:
            max_r = self.field_vars["Maximum Range"].get()
            min_r = self.field_vars["Minimum Range"].get()
            bit_r = self.field_vars["Bit Rate"].get()
            analog_v = self.field_vars["Analog Voltage"].get()

            # Robust type checking and conversion
            max_r = float(max_r) if isinstance(max_r, (int, float)) else float(str(max_r))
            min_r = float(min_r) if isinstance(min_r, (int, float)) else float(str(min_r))
            bit_r = int(bit_r) if isinstance(bit_r, int) else int(str(bit_r))
            analog_v = float(analog_v) if isinstance(analog_v, (int, float)) else float(str(analog_v))

            if bit_r <= 0:
                raise ValueError("Bit Rate must be a positive integer.")
            if max_r <= min_r:
                raise ValueError("Maximum Range must be greater than Minimum Range.")


            self.last_calculated_params = {
                "max_range": max_r,
                "min_range": min_r,
                "bit_rate": bit_r,
                "analog_voltage": analog_v
            }

            quantized_val = compute_quantized_value(max_r, min_r, bit_r, analog_v)

            # Output label will now always be yellow for values too
            self.output_label_value.config(fg=self.quantized_value_color) 
            self.output_value_text.set(f"{quantized_val:.4f} V")

            if not (min_r <= analog_v <= max_r):
                messagebox.showwarning("Input Warning",
                                       f"Analog voltage ({analog_v}V) is outside the specified range ({min_r}V to {max_r}V).\n"
                                       "Quantization has proceeded, but results might be unexpected if this is not intended.")
            
            self._update_quantization_plot()

        except ValueError as e:
            messagebox.showerror("Input Error", f"Please ensure all fields have valid numbers and ranges. Details: {e}")
            self.output_value_text.set("N/A")
            # Revert output label color to Yellow on error
            self.output_label_value.config(fg=self.na_color) 
            self.last_calculated_params = None
            self._update_quantization_plot()
        except Exception as e:
            messagebox.showerror("Calculation Error", f"An unexpected error occurred during quantization: {e}")
            self.output_value_text.set("Error")
            # Revert output label color to Yellow on error
            self.output_label_value.config(fg=self.na_color) 
            self.last_calculated_params = None
            self._update_quantization_plot()

    def _clear_inputs(self):
        for text_field in self.field_vars:
            if text_field == "Bit Rate":
                self.field_vars[text_field].set(8) # Default bit rate
            elif text_field == "Maximum Range":
                self.field_vars[text_field].set(10.0) # Default max range
            elif text_field == "Minimum Range":
                self.field_vars[text_field].set(-10.0) # Default min range
            elif text_field == "Analog Voltage":
                self.field_vars[text_field].set(5.0) # Default analog voltage

        self.output_value_text.set("N/A")
        # Revert output label color to Yellow when cleared to N/A
        self.output_label_value.config(fg=self.na_color) 
        self._hide_adjustment_buttons()
        self._hide_output_analysis_tooltip(None)
        self._hide_field_tooltip()
        self.last_calculated_params = None
        self._update_quantization_plot()

    def _on_window_configure(self, event):
        """
        Handles window configuration events to control About button visibility.
        """
        # Only act if the event is from the root window itself and not its children
        if event.widget == self.master:
            current_state = self.master.state()
            if current_state == 'zoomed': # Maximized state
                self.about_button.grid(row=0, column=6, padx=5, pady=5, sticky="ne")
            else: # Normal or iconic (minimized)
                self.about_button.grid_forget()


# --- Main execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = QuantizerApp(root)
    root.mainloop()