import time as Time
import subprocess
from bokeh.io import export_png, curdoc
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import main
from tkinter import filedialog
import plots
import os
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk



start = Time.time()
def toggle_richards_parameter(*args):
    # Enable/disable the entry field based on the selected model
    if model_combo.get() == "richards":
        m_entry.config(state="normal")
    else:
        m_entry.delete(0, "end")
        m_entry.config(state="disabled")
def define_distribution():
    try:
        volume = float(volume_entry.get())
        mean = float(mean_entry.get())
        std = float(std_entry.get())
        distribution = distribution_combo.get()
        data = main.create_data_by_volume(mean, std, volume, distribution)
        bacteria_pool=int(bacteria_pool_entry.get())
        r=float(r_entry.get())
        time=int(time_entry.get())
        time_step=float(time_step_entry.get())
        model=model_combo.get()
        if model=='richards':
            m=float(m_entry.get())
        else:
            m=None
        df=main.distribute_bacteria(bacteria_pool,data)
        dic,plot=plots.graph_panel(df,volume,time,mean,std,distribution,bacteria_pool, r, model,time_step, m)
        desktop_path = str(Path.home() / "Desktop")
        save_dfs_to_csv(dic,df,time_step,plot,directory=os.path.join(desktop_path, "raw_data"))
    except ValueError:
        result_label.config(text="Please enter valid numbers")


def save_dfs_to_csv(dic,df,time_step,plot, directory):
    # Generate filename with runtime
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    subdirectory = os.path.join(directory, now)

    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    # Save df as CSV
    raw_data_filename = f"raw_data_{now}.txt"
    raw_data_filepath = os.path.join(subdirectory, raw_data_filename)
    df.to_csv(raw_data_filepath,sep='\t', index=False)
    transposed_dfs = [df.drop(columns=['time', 'K', 'droplet size']) for df in dic.values()]
    transposed_dfs = [df.transpose() for df in transposed_dfs]
    concatenated_df = pd.concat(transposed_dfs, ignore_index=True)
    concatenated_df.columns=concatenated_df.columns.astype(int)
    concatenated_df.columns=(concatenated_df.columns*time_step).round(2)
    concatenated_filename = f"concatenated_data_{now}.txt"
    concatenated_filepath = os.path.join(subdirectory, concatenated_filename)
    concatenated_df.to_csv(concatenated_filepath,sep='\t', index=False)
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    driver = webdriver.Chrome(options=chrome_options)
    for i, plot in enumerate(plot):
        export_png(plot, filename=os.path.join(subdirectory, f"plot_{i}.png"), webdriver=driver, height=1500,width=1500)
    driver.quit()  # Close the webdriver after saving all plots
    result_label.config(text=f"CSV files saved: {raw_data_filename}, {concatenated_filename}")
def upload_csv():
    time=int(time_entry.get())
    r=float(r_entry.get())
    model=model_combo.get()
    time_step = float(time_step_entry.get())
    if model=='richards':
        m=float(m_entry.get())
    else:
        m=None
    distribution=None
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        # Check if the DataFrame contains the required columns
    if 'droplet size' in df.columns and 'num of bacteria' in df.columns:
        df = df.sort_values(by='droplet size', ascending=True)
        df=df.reset_index(drop=True)
        result_label.config(text="CSV file uploaded successfully.")
        df['initial density']=(df['num of bacteria']/df['droplet size']).astype(float)
        valid_density_rows = df['initial density'] > 0  # find the rows with valid density
        df.loc[valid_density_rows, 'Rs'] = -5.1 - 0.0357*np.log2(df['droplet size']) - 0.81*np.log2(df['initial density'])  # calculate the Rs value for each droplet
        df.loc[~valid_density_rows, 'Rs'] = 0
        df['k'] = (df['num of bacteria'] * (2 ** df['Rs'])).astype(int)  # calculate the k value for each droplet
        df.loc[df['num of bacteria'] > df['k'], 'k'] = df.loc[df['num of bacteria'] > df['k'], 'num of bacteria'].astype(int)  # if the number of bacteria is greater than the k value, set the k value to the number of bacteria
        df['bin'] = np.log10(df['droplet size'])  # create a new column to store the log10 of the droplet size
        dic,plot=plots.graph_panel(df,df['droplet size'].sum(),time,df['droplet size'].mean(),df['droplet size'].std(),distribution, df['num of bacteria'].sum(), r,model,time_step,m)
        desktop_path = str(Path.home() / "Desktop")
        save_dfs_to_csv(dic, df, plot, directory=os.path.join(desktop_path, "raw_data"))
    else:
        result_label.config(text="The uploaded CSV file does not contain the required columns.")
def open_scanning_gui():
    subprocess.Popen(["python", "scanning.py"])
if __name__ == "__main__":
    # Create main application window
    root = tk.Tk()
    root.title("Growth Rate Simulator")
    # Create input fields
    input_frame = ttk.Frame(root, style="My.TFrame")
    input_frame.grid(row=0, column=0, padx=10, pady=10)

    style = ttk.Style()
    style.configure("My.TFrame", background="light gray")

    # Add a title to the frame
    title_label = ttk.Label(input_frame, text="Droplets Features", font=("TkDefaultFont", 12, "bold"))
    title_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")  # Align to the left

    distribution_label = ttk.Label(input_frame, text="Select Distribution:")
    distribution_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
    distribution_options = ["lognormal", "truncated normal", "uniform"]
    distribution_combo = ttk.Combobox(input_frame, values=distribution_options)
    distribution_combo.current(0)  # Set default selection
    distribution_combo.grid(row=1, column=1, padx=5, pady=5)
    distribution_combo.set("lognormal")

    volume_label = ttk.Label(input_frame, text="Volume (microLiter):")
    volume_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
    volume_entry = ttk.Entry(input_frame)
    volume_entry.grid(row=2, column=1, padx=5, pady=5)
    volume_entry.insert(0, "100000000")  # Set default volume


    mean_label = ttk.Label(input_frame, text="Mean (microLiter):")
    mean_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
    mean_entry = ttk.Entry(input_frame)
    mean_entry.grid(row=3, column=1, padx=5, pady=5)
    mean_entry.insert(0, "3")     # Set default mean


    std_label = ttk.Label(input_frame, text="Standard Deviation (microLiter):")
    std_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
    std_entry = ttk.Entry(input_frame)
    std_entry.grid(row=4, column=1, padx=5, pady=5)
    std_entry.insert(0, "1.5")       # Set default standard deviation


    title_label = ttk.Label(input_frame, text="Bacteria culture features", font=("TkDefaultFont", 12, "bold"))
    title_label.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="w")  # Align to the left

    bacteria_pool_label = ttk.Label(input_frame, text="how many bacteria:")
    bacteria_pool_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")
    bacteria_pool_entry = ttk.Entry(input_frame)
    bacteria_pool_entry.grid(row=6, column=1, padx=5, pady=5)
    bacteria_pool_entry.insert(0, "10000")  # Set default bacteria pool


    r_label = ttk.Label(input_frame, text="growth rate (r):")
    r_label.grid(row=7, column=0, padx=5, pady=5, sticky="w")
    r_entry = ttk.Entry(input_frame)
    r_entry.grid(row=7, column=1, padx=5, pady=5)
    r_entry.insert(0, "0.5")     # Set default growth rate


    title_label = ttk.Label(input_frame, text="Model features", font=("TkDefaultFont", 12, "bold"))
    title_label.grid(row=8, column=0, columnspan=2, padx=5, pady=5, sticky="w")  # Align to the left

    time_label = ttk.Label(input_frame, text="how many hours:")
    time_label.grid(row=9, column=0, padx=5, pady=5, sticky="w")
    time_entry = ttk.Entry(input_frame)
    time_entry.grid(row=9, column=1, padx=5, pady=5)
    time_entry.insert(0, "48")    # Set default time


    time_step_label = ttk.Label(input_frame, text="time steps size:")
    time_step_label.grid(row=10, column=0, padx=5, pady=5, sticky="w")
    time_step_entry = ttk.Entry(input_frame)
    time_step_entry.grid(row=10, column=1, padx=5, pady=5)
    time_step_entry.insert(0, "0.1")      # Set default time step


    model_label = ttk.Label(input_frame, text="Select model:")
    model_label.grid(row=11, column=0, padx=5, pady=5, sticky="w")
    model_options = ["logistic", "gompertz", "richards"]
    model_combo = ttk.Combobox(input_frame, values=model_options)
    model_combo.current(0)  # Set default selection
    model_combo.grid(row=11, column=1, padx=5, pady=5)
    model_combo.bind("<<ComboboxSelected>>", toggle_richards_parameter)  # Correctly bind the function
    model_combo.set("logistic")



    m_label = ttk.Label(input_frame, text="m (for richards model):")  # Label for richards parameter
    m_label.grid(row=12, column=0, padx=5, pady=5, sticky="w")
    m_entry = ttk.Entry(input_frame, state="disabled")  # Entry field for richards parameter, initially disabled
    m_entry.grid(row=12, column=1, padx=5, pady=5)

    # Create button to calculate distribution
    calculate_button = ttk.Button(root, text="Calculate", command=define_distribution)
    calculate_button.grid(row=13, column=0, columnspan=2, padx=5, pady=5)

    upload_button = ttk.Button(root, text="Upload CSV", command=upload_csv)
    upload_button.grid(row=14, column=0, columnspan=2, padx=5, pady=5)

    # Create label to display result
    result_label = ttk.Label(root, text="")
    result_label.grid(row=15, column=0, columnspan=2, padx=5, pady=5)

    open_scanning_button = ttk.Button(root, text="Open Scanning GUI", command=open_scanning_gui)
    open_scanning_button.grid(row=16, column=0, columnspan=2, padx=5, pady=5)
    def on_closing():
        root.destroy()

    # Configure the closing event
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Run the application
    root.mainloop()
