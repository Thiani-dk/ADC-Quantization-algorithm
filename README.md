# Analog to Digital Quantizer

This is a Python-based GUI application that visualizes the analog-to-digital quantization process. It allows users to adjust parameters like maximum range, minimum range, bit rate, and analog voltage, and see the quantized output both numerically and on a real-time plot.

## Features

* **Adjustable Parameters:** Set Maximum Range, Minimum Range, Bit Rate, and Analog Voltage via input fields, sliders, and increment/decrement buttons.
* **Real-time Quantization Plot:** Dynamically updates a Matplotlib plot to show quantization levels and how an analog signal is digitized.
* **Quantized Output:** Displays the numerically quantized voltage.
* **Interactive Tooltips:** Hover over the output for detailed feedback on quantization parameters and their implications for sound quality.
* **About Section:** Provides information about the application and basic usage instructions.
* **Dynamic UI:** Adjustment buttons and info buttons appear contextually when an input field is selected.

## Prerequisites

To run this application, you need:

* **Python 3.x**
* **pip** (Python package installer, usually comes with Python)

## Dependencies

The following Python libraries are required:

* `tkinter` (Tkinter is typically included with standard Python installations, but some Linux distributions might require it to be installed separately.)
* `numpy`
* `matplotlib`

## Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
    (Replace `<your-repo-url>` and `<your-repo-name>` with your actual repository details.)

2.  **Install the required Python libraries:**
    Open your terminal or command prompt and run:
    ```bash
    pip install numpy matplotlib
    ```

    **Note for Linux users regarding Tkinter:**
    If you encounter an error related to `tkinter` not being found, you might need to install it at the system level. For Debian/Ubuntu-based systems, use:
    ```bash
    sudo apt-get update
    sudo apt-get install python3-tk
    ```
    For Fedora/RHEL-based systems, use:
    ```bash
    sudo dnf install python3-tkinter
    # or for older yum
    sudo yum install python3-tkinter
    ```
    For other operating systems, please refer to your distribution's documentation for installing Tkinter.

## How to Run

1.  Navigate to the directory where you have saved the `quantizer_app.py` (or whatever you named your main Python file).
    ```bash
    cd /path/to/your/quantizer_app
    ```

2.  Run the application using Python:
    ```bash
    python quantizer_app.py
    ```

    This will open the Analog to Digital Quantizer GUI.

## Usage

* Adjust the "Maximum Range", "Minimum Range", "Bit Rate", and "Analog Voltage" values using the input boxes, sliders, or the `+` and `-` buttons that appear when an input box is focused.
* The plot on the right will update in real-time to show the quantization levels and how the analog signal is being sampled.
* Click the "Quantize" button to see the precise quantized output for the current "Analog Voltage".
* Hover over the "Quantized Voltage" output to get detailed feedback and theoretical sound quality analysis.
* Click the `?` button next to an input field for an explanation of that parameter.
* Use the toolbar below the plot to pan, zoom, or save the graph as an image.
* Click "Clear" to reset all input values to their defaults.
* The "About" button (visible when the window is maximized) provides general information and usage tips.

---
