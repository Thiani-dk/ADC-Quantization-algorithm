import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # Import ttk for themed widgets

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
        raise ValueError("Maximum Range must be greater than Minimum Range.")
    if bit_rate <= 0:
        raise ValueError("Bit Rate must be a positive integer.")

    # The calculation will still proceed even if analog_voltage is out of range
    # The warning will be shown in the GUI
    # if not (min_range <= analog_voltage <= max_range):
    #     pass

    num_quantization_levels = 2**bit_rate
    # Avoid division by zero if only one quantization level (bit_rate=0, though we enforce bit_rate > 0)
    # or if max_range == min_range.
    if num_quantization_levels <= 1 or (max_range - min_range) == 0:
        # If there's effectively only one level or no range, all values map to min_range
        # or handle as an error based on your desired behavior for such edge cases.
        # For now, let's make it map to min_range as the only representable value.
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
        
        # New: Info button colors
        self.info_button_fg_color = "#FFFF00" # Yellow for the question mark
        self.info_button_active_fg_color = "#FFA500" # Orange for hover

        # NEW: Quantized output specific color
        self.quantized_value_color = "#FFFF00" # Bright Yellow for the output number and 'V'

        self.title_font = ("Arial", 18, "bold")
        self.label_font = ("Arial", 11)
        self.entry_font = ("Consolas", 11) # Monospaced font for numbers
        self.button_font = ("Arial", 10, "bold")
        self.output_label_font = ("Arial", 13, "bold") # General output label font
        self.output_value_font = ("Consolas", 14, "bold") # Larger, bold, monospaced for the value
        self.tooltip_font = ("Arial", 9)
        self.field_tooltip_font = ("Arial", 9)
        self.close_button_font = ("Arial", 8) # Smaller font for close button

        # Configure the root window background
        self.master.configure(bg=self.bg_color)

        # Dictionary to hold Entry widgets for easy access
        self.entry_fields = {}
        self.current_focused_entry = None
        self.tooltip_window = None # To hold the output tooltip Toplevel window
        self.field_tooltip_window = None # To hold the field explanation tooltip Toplevel window

        # --- Explanations for each input field ---
        self.field_explanations = {
            "Maximum Range": "This defines the highest voltage value that your analog-to-digital converter (ADC) can measure. Setting it too low for a given signal can lead to clipping (distortion) if the signal exceeds this range.",
            "Minimum Range": "This defines the lowest voltage value that your analog-to-digital converter (ADC) can measure. Setting it too high for a given signal can also lead to clipping or a loss of lower amplitude information if the signal falls below this range.",
            "Bit Rate": "The 'Bit Rate' (or bit depth) determines the number of quantization levels available. More bits mean more levels, leading to finer resolution and less quantization error (noise), which improves the quality of the digitized signal. For example, 8 bits give 256 levels, while 16 bits give 65,536 levels. Lower bit rates can result in a 'grainy' or 'noisy' sound due to larger quantization steps.",
            "Analog Voltage": "This is the instantaneous voltage of the analog signal that you want to convert into a digital value. The ADC will approximate this voltage to the nearest available quantization level within the defined range."
        }

        # --- Main Frame for layout and border ---
        self.main_frame = tk.Frame(master, padx=20, pady=20, bd=2, relief="groove", bg=self.frame_bg_color, highlightbackground=self.text_color, highlightthickness=1)
        self.main_frame.pack(pady=20, padx=20)

        # --- Title Label ---
        # Changed columnspan to 6 to accommodate +,-,? buttons
        self.title_label = tk.Label(self.main_frame, text="Analog to Digital Quantizer", font=self.title_font, fg=self.text_color, bg=self.frame_bg_color)
        self.title_label.grid(row=0, column=0, columnspan=6, pady=(0, 20))

        # --- Input Fields and Labels ---
        input_specs = [
            ("Maximum Range", "V"),
            ("Minimum Range", "V"),
            ("Bit Rate", "Bit"),
            ("Analog Voltage", "V")
        ]

        row_num = 1
        for text, unit in input_specs:
            tk.Label(self.main_frame, text=f"{text}:", font=self.label_font, fg=self.text_color, bg=self.frame_bg_color).grid(row=row_num, column=0, sticky="w", pady=5)

            entry = tk.Entry(self.main_frame, width=20, bd=1, relief="solid",
                             bg=self.entry_bg_color, fg=self.entry_fg_color,
                             insertbackground=self.entry_fg_color, # Cursor color
                             font=self.entry_font)
            entry.grid(row=row_num, column=1, pady=5, padx=5)

            tk.Label(self.main_frame, text=unit, font=self.label_font, fg=self.text_color, bg=self.frame_bg_color).grid(row=row_num, column=2, sticky="w")

            self.entry_fields[text] = entry
            entry.bind("<FocusIn>", self._on_entry_focus_in)
            # No permanent info button here
            row_num += 1

        # --- Configure Button Styles for ttk ---
        style = ttk.Style()
        style.theme_use('clam') # 'clam' is a good modern-looking theme base

        # General button style (used for + and -)
        style.configure("TButton",
                        background=self.button_bg_color,
                        foreground=self.button_fg_color,
                        font=self.button_font,
                        padding=5,
                        relief="flat") # Flat button
        style.map("TButton",
                  background=[('active', self.button_active_bg_color)], # Color on hover
                  foreground=[('active', self.button_fg_color)]) # Text color on hover

        # Specific style for Quantize button
        style.configure("Quantize.TButton",
                        background=self.quantize_button_color,
                        foreground=self.button_fg_color)
        style.map("Quantize.TButton",
                  background=[('active', self.quantize_button_hover_color)],
                  foreground=[('active', self.button_fg_color)])

        # Specific style for Clear Selection button
        style.configure("Clear.TButton",
                        background=self.clear_button_color,
                        foreground=self.button_fg_color)
        style.map("Clear.TButton",
                  background=[('active', self.clear_button_hover_color)],
                  foreground=[('active', self.button_fg_color)])

        # Small adjustment button style (+/-)
        style.configure("Small.TButton",
                        font=("Arial", 9, "bold"),
                        width=4,
                        padding=2)
        style.map("Small.TButton",
                  background=[('active', self.button_active_bg_color)],
                  foreground=[('active', self.button_fg_color)])
        
        # --- NEW: Info button style (transparent background, yellow text) ---
        style.configure("Info.TButton",
                        background=self.frame_bg_color, # Match frame background for 'invisible' box
                        foreground=self.info_button_fg_color, # Yellow text
                        font=("Arial", 12, "bold"), # Slightly larger for the '?'
                        width=3, # Smaller width
                        padding=0, # No padding for invisible box effect
                        relief="flat") # Flat relief
        style.map("Info.TButton",
                  background=[('active', self.frame_bg_color)], # Keep background invisible on hover
                  foreground=[('active', self.info_button_active_fg_color)]) # Orange on hover


        # --- Quantize Button ---
        # Adjusted columnspan to accommodate 3 dynamic buttons (+,-,?)
        self.quantize_button = ttk.Button(self.main_frame, text="Quantize", command=self._quantize, width=15, style="Quantize.TButton")
        self.quantize_button.grid(row=row_num, column=3, columnspan=3, pady=20, padx=10, sticky="e") # Spans 3 columns

        # --- Clear Selection Button ---
        self.clear_button = ttk.Button(self.main_frame, text="Clear Selection", command=self._clear_inputs, width=15, style="Clear.TButton")
        self.clear_button.grid(row=row_num + 1, column=3, columnspan=3, pady=5, padx=10, sticky="e") # Spans 3 columns

        # --- Output Section ---
        # This section will hold two labels for separate coloring
        self.output_frame = tk.Frame(self.main_frame, bg=self.frame_bg_color)
        self.output_frame.grid(row=row_num + 2, column=0, columnspan=6, pady=20, sticky="w")

        # Label for the static "Quantized Voltage:" text
        self.output_label_prefix = tk.Label(self.output_frame, text="Quantized Voltage: ",
                                            font=self.output_label_font,
                                            fg=self.text_color, bg=self.frame_bg_color)
        self.output_label_prefix.pack(side=tk.LEFT, padx=(0,0)) # No right padding

        # Label for the dynamic value and unit (will be yellow)
        self.output_value_text = tk.StringVar()
        self.output_value_text.set("N/A") # Default value
        self.output_label_value = tk.Label(self.output_frame, textvariable=self.output_value_text,
                                           font=self.output_value_font,
                                           fg=self.quantized_value_color, # Yellow color for this part
                                           bg=self.frame_bg_color)
        self.output_label_value.pack(side=tk.LEFT)

        # --- Bind hover events to the output frame for the tooltip ---
        self.output_frame.bind("<Enter>", self._show_output_analysis_tooltip)
        self.output_frame.bind("<Leave>", self._hide_output_analysis_tooltip)
        # Also bind children labels, so hover works over the whole output text
        self.output_label_prefix.bind("<Enter>", self._show_output_analysis_tooltip)
        self.output_label_prefix.bind("<Leave>", self._hide_output_analysis_tooltip)
        self.output_label_value.bind("<Enter>", self._show_output_analysis_tooltip)
        self.output_label_value.bind("<Leave>", self._hide_output_analysis_tooltip)


        # --- Increment/Decrement and Info Buttons (dynamic) ---
        self.inc_button = ttk.Button(self.main_frame, text="+", command=lambda: self._adjust_value(1), style="Small.TButton")
        self.dec_button = ttk.Button(self.main_frame, text="-", command=lambda: self._adjust_value(-1), style="Small.TButton")
        self.info_button = ttk.Button(self.main_frame, text="?", command=self._trigger_field_tooltip, style="Info.TButton")

        # Initially hide them
        self.inc_button.grid_forget()
        self.dec_button.grid_forget()
        self.info_button.grid_forget()

        self.master.bind("<Button-1>", self._on_window_click)


    # --- New: Helper to trigger field tooltip from info button ---
    def _trigger_field_tooltip(self):
        """Triggers the field tooltip for the currently focused entry."""
        if self.current_focused_entry:
            # Find the field name corresponding to the current focused entry
            field_name = None
            for name, entry_widget in self.entry_fields.items():
                if entry_widget == self.current_focused_entry:
                    field_name = name
                    break
            if field_name:
                self._show_field_tooltip(field_name, self.current_focused_entry)


    # --- Modified Method for Field-Specific Tooltips (now triggered by Info Button) ---
    def _show_field_tooltip(self, field_name, entry_widget):
        """Displays a tooltip explanation for a specific input field."""
        if self.field_tooltip_window: # Hide any existing field tooltip
            self._hide_field_tooltip()

        if not field_name or field_name not in self.field_explanations:
            return

        tooltip_message = self.field_explanations[field_name]

        # Get coordinates of the entry widget to position the tooltip
        x = entry_widget.winfo_rootx()
        y = entry_widget.winfo_rooty()
        
        # Calculate position for tooltip (relative to the entry field)
        tooltip_x = x + entry_widget.winfo_width() + 60 # Increased offset
        tooltip_y = y - 10 # Slightly above the entry field

        self.field_tooltip_window = tk.Toplevel(self.master)
        self.field_tooltip_window.wm_overrideredirect(True)
        self.field_tooltip_window.wm_geometry(f"+{tooltip_x}+{tooltip_y}")

        # Frame to hold the label and the close button
        tooltip_frame = tk.Frame(self.field_tooltip_window,
                                 background=self.field_tooltip_bg,
                                 relief="solid", borderwidth=1)
        tooltip_frame.pack(padx=0, pady=0)

        label = tk.Label(tooltip_frame, text=tooltip_message,
                         background=self.field_tooltip_bg, foreground=self.field_tooltip_fg,
                         font=self.field_tooltip_font,
                         justify=tk.LEFT,
                         padx=7, pady=5,
                         wraplength=280) # Wrap text for better readability
        label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Add a Close button to the field tooltip ---
        close_button = ttk.Button(tooltip_frame, text="Close",
                                  command=self._hide_field_tooltip,
                                  style="TButton") # Use general TButton style
        close_button.pack(side=tk.BOTTOM, pady=5, padx=5)

    def _hide_field_tooltip(self):
        """Hides the field explanation tooltip."""
        if self.field_tooltip_window:
            self.field_tooltip_window.destroy()
            self.field_tooltip_window = None

    # --- Renamed and Enhanced Method for Output Analysis Tooltip ---
    def _show_output_analysis_tooltip(self, event):
        """Displays a tooltip text box with analysis of the quantized output."""
        if self.tooltip_window: # Don't show tooltip if already visible
            return

        current_output_str = self.output_value_text.get().strip() # Get from output_value_text
        
        # Only show tooltip if there's a valid numerical output and calculation params exist
        if not current_output_str or current_output_str in ["N/A", "Error"] or self.last_calculated_params is None:
            return

        tooltip_message = "" # Start with an empty message for structured content
        
        # --- Dynamic commentary based on quantization parameters ---
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
                if bit_r <= 7: # Very low bit rates, likely to have significant noise
                    tooltip_message += "- With very low bit rates (e.g., 1-7 bits), quantization steps are very large. This can lead to severe 'quantization noise' and a highly 'steppy' or 'grainy' sound, particularly noticeable in quiet passages. The digital representation will have a very limited dynamic range, resulting in a 'crude' or 'lo-fi' audio feel. Imagine an extremely basic digital recording.\n"
                elif 8 <= bit_r <= 11: # Medium-low, still perceptible noise
                    tooltip_message += "- At this bit rate, quantization noise is still quite perceptible, especially on quiet sounds or decays. The audio may lack smoothness and sound somewhat 'harsh' or 'digital'. It's often used for speech or less critical audio where file size is paramount (e.g., telephone audio or older multimedia presentations).\n"
                elif 12 <= bit_r <= 15: # Good for general use, but not "audiophile"
                    tooltip_message += "- This range offers a good balance. Quantization noise is significantly reduced and often imperceptible to the average listener for most audio. The sound should be clear and generally pleasant, though critical listening might reveal minor imperfections in very high-fidelity recordings. This would be typical for many streaming services or good quality compressed audio.\n"
                else: # 16 bits or higher (CD quality and beyond)
                    tooltip_message += "- With 16 bits or more, the quantization noise is extremely low and generally below the threshold of human hearing (the 'noise floor'). This results in a very smooth, accurate, and high-fidelity digital representation. The sound should be clean, detailed, and have excellent dynamic range, comparable to professional audio recordings or CD quality (16-bit, 44.1 kHz).\n"
                    
                # Check for clipping (if analog voltage is out of range)
                if not (min_r <= analog_v <= max_r):
                    tooltip_message += "\n*Warning: Input Clipping*\n" # Subheading for warning
                    tooltip_message += "- The input analog voltage was outside the specified range. This would result in 'clipping' during conversion, leading to severe audible distortion at the peaks of the signal (flat-topping or flat-bottoming). This sounds like a harsh, fuzzy distortion, especially on loud sounds.\n"

            else: # If only one level or invalid range (e.g., bit_rate = 0, or max_r == min_r)
                tooltip_message += "**Quantization Feedback:**\n"
                tooltip_message += "- With very few bits or an invalid voltage range, the system cannot accurately represent a varying analog signal. The output will likely be highly distorted, clipped, or simply a constant value, leading to severe loss of audio information. The resulting 'sound' would be unusable.\n"

        except (ValueError, KeyError): # Handle cases where last_calculated_params might be incomplete or invalid
            tooltip_message += "\n(Cannot provide detailed feedback due to missing or invalid input values from the last calculation.)\n"
        except Exception as e:
            tooltip_message += f"\n(Error generating detailed feedback: {e})\n"

        # Remove trailing newlines from the message for cleaner presentation
        tooltip_message = tooltip_message.strip()

        # Create a Toplevel window for the tooltip
        self.tooltip_window = tk.Toplevel(self.master)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{event.x_root + 15}+{event.y_root + 15}")

        label = tk.Label(self.tooltip_window, text=tooltip_message,
                         background=self.tooltip_bg, foreground=self.tooltip_fg,
                         relief="solid", borderwidth=1,
                         font=self.tooltip_font,
                         justify=tk.LEFT,
                         padx=10, pady=7, # Increased padding for better appearance
                         wraplength=400) # Increased wraplength for more content
        label.pack()

    def _hide_output_analysis_tooltip(self, event):
        """Hides the output analysis tooltip text box."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def _on_entry_focus_in(self, event):
        """Called when an Entry widget gains focus."""
        self._hide_field_tooltip()
        self.current_focused_entry = event.widget
        grid_info = self.current_focused_entry.grid_info()
        row = grid_info['row']

        # --- Grid the +, -, and ? buttons dynamically ---
        self.inc_button.grid(row=row, column=3, padx=2, sticky="w") # Column 3
        self.dec_button.grid(row=row, column=4, padx=2, sticky="w") # Column 4
        self.info_button.grid(row=row, column=5, padx=2, sticky="w") # Column 5

    def _on_window_click(self, event):
        """Called when a click occurs anywhere on the main window."""
        is_entry = isinstance(event.widget, tk.Entry)
        is_inc_button = (event.widget == self.inc_button.winfo_containing(event.x_root, event.y_root))
        is_dec_button = (event.widget == self.dec_button.winfo_containing(event.x_root, event.y_root))
        is_info_button = (event.widget == self.info_button.winfo_containing(event.x_root, event.y_root))

        # Check if the click was on any tooltip window itself
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

        # If click is not on entry, adjustment buttons, or any tooltip, hide buttons and tooltips
        if not (is_entry or is_inc_button or is_dec_button or is_info_button or is_tooltip_click):
            self._hide_adjustment_buttons()
            self._hide_output_analysis_tooltip(event)
            self._hide_field_tooltip()
            self.master.focus_set()


    def _hide_adjustment_buttons(self):
        """Hides the increment/decrement and info buttons."""
        if self.inc_button.winfo_ismapped():
            self.inc_button.grid_forget()
        if self.dec_button.winfo_ismapped():
            self.dec_button.grid_forget()
        if self.info_button.winfo_ismapped():
            self.info_button.grid_forget()
        self.current_focused_entry = None


    def _adjust_value(self, delta):
        """Adjusts the numeric value of the currently focused entry."""
        if self.current_focused_entry:
            try:
                current_value_str = self.current_focused_entry.get()

                is_bit_rate_field = False
                for field_name, entry_widget in self.entry_fields.items():
                    if entry_widget == self.current_focused_entry and field_name == "Bit Rate":
                        is_bit_rate_field = True
                        break

                if is_bit_rate_field:
                    current_value = int(current_value_str) if current_value_str else 0
                    new_value = current_value + delta
                    if new_value < 1:
                        new_value = 1
                    self.current_focused_entry.delete(0, tk.END)
                    self.current_focused_entry.insert(0, str(int(new_value)))
                else:
                    current_value = float(current_value_str) if current_value_str else 0.0
                    new_value = current_value + delta
                    self.current_focused_entry.delete(0, tk.END)
                    self.current_focused_entry.insert(0, f"{new_value:.1f}")

            except ValueError:
                messagebox.showerror("Input Error", "Please enter a valid number in the selected field before adjusting.")


    def _quantize(self):
        """Retrieves input values, computes quantized value, and displays it."""
        # Hide any active tooltips when a calculation is performed or cleared
        self._hide_output_analysis_tooltip(None)
        self._hide_field_tooltip()
        try:
            max_r = float(self.entry_fields["Maximum Range"].get())
            min_r = float(self.entry_fields["Minimum Range"].get())
            bit_r = int(self.entry_fields["Bit Rate"].get())
            analog_v = float(self.entry_fields["Analog Voltage"].get())

            # Store these values for the tooltip analysis
            self.last_calculated_params = {
                "max_range": max_r,
                "min_range": min_r,
                "bit_rate": bit_r,
                "analog_voltage": analog_v
            }

            quantized_val = compute_quantized_value(max_r, min_r, bit_r, analog_v)

            # Update the separate output_label_value
            self.output_value_text.set(f"{quantized_val:.4f} V")

            if not (min_r <= analog_v <= max_r):
                messagebox.showwarning("Input Warning",
                                       f"Analog voltage ({analog_v}V) is outside the specified range ({min_r}V to {max_r}V).\n"
                                       "Quantization has proceeded, but results might be unexpected if this is not intended.")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Please ensure all fields have valid numbers. Details: {e}")
            self.output_value_text.set("N/A") # Set only the value part
            self.last_calculated_params = None # Clear params if calculation failed
        except Exception as e:
            messagebox.showerror("Calculation Error", f"An unexpected error occurred during quantization: {e}")
            self.output_value_text.set("Error") # Set only the value part
            self.last_calculated_params = None # Clear params if calculation failed

    def _clear_inputs(self):
        """Clears all input fields and the output display."""
        for entry in self.entry_fields.values():
            entry.delete(0, tk.END)
        self.output_value_text.set("N/A") # Clear only the value part
        self._hide_adjustment_buttons()
        self._hide_output_analysis_tooltip(None)
        self._hide_field_tooltip()
        self.last_calculated_params = None


# --- Main execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = QuantizerApp(root)
    root.mainloop()