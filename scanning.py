import glob
import os
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk, filedialog
from tkinter import messagebox
import main
from multiprocessing import Pool
from itertools import product
import itertools
import pandas as pd
import numpy as np
from time import time as T
import scanning_analysis


def save_df_to_csv(df_arg_tuple, run_directory):
    df, arg,kind = df_arg_tuple
    filename = f"{arg} {kind}.txt"
    filepath = os.path.join(run_directory, filename)
    df.to_csv(filepath,sep='\t', index=False)


def process_arguments_scanning(arg):
    org_arg=arg
    arg = arg[:-1]
    iter = T()
    if len(arg) == 9:
        m = None
        distribution, volume, mean, std, pool_size, r, time, max_step, model = arg
    else:
        distribution, volume, mean, std, pool_size, r, time, max_step, model, m = arg
    waterscape = main.create_data_by_volume(mean, std, volume, distribution)
    print(f'Creating waterscape took {T() - iter} seconds')
    iter = T()
    bac = main.distribute_bacteria(pool_size, waterscape)
    print(f'Distributing bacteria took {T() - iter} seconds')
    iter = T()
    sim = main.simulation_full_waterscape(time, bac, model, r, max_step, m)
    print(f'Simulation took {T() - iter} seconds')
    list_of_dirs = glob.glob(os.path.join(os.path.expanduser('~'), 'Desktop', 'raw_data', 'scanning_tool', '*'))
    run_directory = max(list_of_dirs, key=os.path.getctime)
    raw_data = (bac, org_arg, "raw_data")
    transposed_dfs = [df.drop(columns=['time', 'K', 'droplet size']) for df in sim.values()]
    transposed_dfs = [df.transpose() for df in transposed_dfs]
    concatenated_data = pd.concat(transposed_dfs, ignore_index=True)
    concatenated_data.columns = concatenated_data.columns.astype(int)
    concatenated_data.columns = (concatenated_data.columns*max_step).round(2)
    concatenated_data = (concatenated_data, org_arg, "simulation")
    save_df_to_csv(raw_data, run_directory)
    save_df_to_csv(concatenated_data, run_directory)


def scanning(*args):
    combination,original_combinations = args[:len(args)//2],args[len(args)//2:]
    start = T()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_directory = os.path.join(os.path.expanduser('~'), 'Desktop', 'raw_data', 'scanning_tool', timestamp)
    os.makedirs(run_directory, exist_ok=True)
    with Pool() as pool:
        pool.map(process_arguments_scanning, combination)
    print(f'Finished processing arguments in {T() - start} seconds')


def process_arguments_fixed_waterscape(arg):
    org_arg=arg[1:]
    arg = arg[:-1]
    start = T()
    if len(arg) == 6:
        m = None
        data,pool_size, r, time, max_step, model = arg
    else:
        data, pool_size, r, time, max_step, model, m = arg
    bac = main.distribute_bacteria(pool_size, data)
    print(f'Distributing bacteria took {T() - start} seconds')
    sim = main.simulation_full_waterscape(time, bac, model, r, max_step, m)
    print(f'Simulation took {T() - start} seconds')
    list_of_dirs = glob.glob(os.path.join(os.path.expanduser('~'), 'Desktop', 'raw_data', 'scanning_tool', '*'))
    run_directory = max(list_of_dirs, key=os.path.getctime)
    raw_data = (bac, org_arg, "raw_data")
    transposed_dfs = [df.drop(columns=['time', 'K', 'droplet size']) for df in sim.values()]
    transposed_dfs = [df.transpose() for df in transposed_dfs]
    concatenated_data = pd.concat(transposed_dfs, ignore_index=True)
    concatenated_data.columns = concatenated_data.columns.astype(int)
    concatenated_data.columns = (concatenated_data.columns*max_step).round(2)
    concatenated_data = (concatenated_data, org_arg, "simulation")
    save_df_to_csv(raw_data, run_directory)
    save_df_to_csv(concatenated_data, run_directory)

def fixed_waterscape(*args):
    combination,original_combinations = args[:len(args)//2],args[len(args)//2:]
    start = T()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_directory = os.path.join(os.path.expanduser('~'), 'Desktop', 'raw_data', 'scanning_tool', timestamp)
    os.makedirs(run_directory, exist_ok=True)
    with Pool() as pool:
        pool.map(process_arguments_fixed_waterscape, combination)
    print(f'Finished processing arguments in {T() - start} seconds')

def process_arguments_fixed_bacterial_distribution(arg):
    if 'Chip' in arg[-1]:
        new_arg = arg[:-2]
    else:
        new_arg = arg[:-1]
    org_arg=arg[1:]
    start = T()
    if len(arg)==5:
        m = None
        data, r, time, max_step, model = new_arg
    else:
        data, r, time, max_step, model, m = new_arg
    sim=main.simulation_full_waterscape(time, data, model, r, max_step, m)
    print(f'Simulation took {T() - start} seconds')
    list_of_dirs = glob.glob(os.path.join(os.path.expanduser('~'), 'Desktop', 'raw_data', 'scanning_tool', '*'))
    run_directory = max(list_of_dirs, key=os.path.getctime)
    raw_data = (data, org_arg, "raw_data")
    transposed_dfs = [df.drop(columns=['time', 'K', 'droplet size']) for df in sim.values()]
    transposed_dfs = [df.transpose() for df in transposed_dfs]
    concatenated_data = pd.concat(transposed_dfs, ignore_index=True)
    concatenated_data.columns = concatenated_data.columns.astype(int)
    concatenated_data.columns = (concatenated_data.columns*max_step).round(2)
    concatenated_data = (concatenated_data, org_arg, "simulation")
    save_df_to_csv(raw_data, run_directory)
    save_df_to_csv(concatenated_data, run_directory)


def fixed_bacterial_distribution(*args):
    combination,original_combinations = args[:len(args)//2],args[len(args)//2:]
    start = T()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_directory = os.path.join(os.path.expanduser('~'), 'Desktop', 'raw_data', 'scanning_tool', timestamp)
    os.makedirs(run_directory, exist_ok=True)
    with Pool() as pool:
        pool.map(process_arguments_fixed_bacterial_distribution, combination)
    print(f'Finished processing arguments in {T() - start} seconds')


param_entries = []  # Define param_entries as a global variable
param_type_vars = []
param_options = {
    "Distribution type": ["lognormal", "truncated normal", "uniform"],
    "Model": ["logistic", "gompertz", "richards"]
}
def validate_parameters():
    for param_label, param_type_var, *entry_widgets in param_entries:
        param_name = param_label.cget('text')
        if param_name in ["Distribution type", "Model"]:
            continue  # Skip validation for Distribution type and Model
        param_type = param_type_var.get()
        if param_type == "Dynamic":
            start_value = entry_widgets[0].get()
            end_value = entry_widgets[1].get()
            num_samples = entry_widgets[2].get()
            if start_value == "start value" or not start_value.replace('.', '', 1).isdigit():
                return False, f"Start value is missing or not a number for {param_label.cget('text')}"
            if end_value == "end value" or not end_value.replace('.', '', 1).isdigit():
                return False, f"End value is missing or not a number for {param_label.cget('text')}"
            if num_samples == "number of samples" or not num_samples.isdigit():
                return False, f"Number of samples is missing or not a number for {param_label.cget('text')}"
            start_value = float(start_value)
            end_value = float(end_value)
            if not end_value > start_value:
                return False, f"The end value must be greater than the start value for {param_label.cget('text')}"
    return True, ""

def run_simulation():
    parameters = {}
    valid, param_name, = validate_parameters()
    if not valid:
        tk.messagebox.showerror("Error", param_name)
        return None
    # Retrieve selected parameter values
    for param_label, param_type_var, *other_entries in param_entries:
        param_name = param_label.cget("text")
        param_type = param_type_var.get()
        values = []
        if param_name in ["Distribution type", "Model"]:
            values = [option.cget("text") for option in other_entries if hasattr(option, 'var') and option.var.get()]
        else:
            for entry_widget in other_entries:
                if isinstance(entry_widget, ttk.Checkbutton) and hasattr(entry_widget, 'var'):
                    values.append(entry_widget.var.get())  # For Checkbutton
                elif isinstance(entry_widget, ttk.Entry):
                    values.append(entry_widget.get())  # For Entry

        unwanted_values = {'start value', 'end value', 'number of samples'}
        values = [val for val in values if val not in unwanted_values]
        if param_type == "Static":
            values = values[:1]
        parameters[param_name] = values
    return parameters
def update_param_entry_widgets(param_type_var, *entry_widgets):
    param_type = param_type_var.get()
    if param_type == "Dynamic":
        num_dynamic = sum(1 for entry_widget in param_type_vars if entry_widget.get() == "Dynamic")
        if num_dynamic > 3:
            param_type_var.set("Static")
            for entry_widget in entry_widgets[1:]:
                entry_widget.grid_remove()
        else:
            for entry_widget in entry_widgets:
                entry_widget.grid()
    else:  # If mode is Static, show all entry widgets
        for entry_widget in entry_widgets:
            entry_widget.grid()
        # Ensure only one entry widget is displayed for parameters in static mode
        if len(entry_widgets) > 1:
            for entry_widget in entry_widgets[1:]:
                entry_widget.grid_remove()
    for param_entry, param_type_var, *other_entries in param_entries:
        param_name = param_entry.cget("text")
        if param_name in ["Distribution type", "Model"]:
            for entry_widget in other_entries:
                entry_widget.grid()
                entry_widget.configure(state='normal')

def csv_to_raw_data(grouped, run_directory):
    for name, group in grouped:
        filtered_data = group[group['time'] == 0]
        raw_data = filtered_data[['Volume', 'Count', 'log_Volume']].copy()
        raw_data.rename(columns={'Volume': 'droplet size', 'Count': 'num of bacteria', 'log_Volume': 'bin'},
                        inplace=True)
        raw_data = raw_data.reset_index(drop=True)
        raw_data.to_csv(os.path.join(run_directory, f'{name} experimental raw_data.txt'), sep='\t', index=False)
def csv_to_simulation_data(df, run_directory):
    grouped = df.groupby('Slice')
    for name, group in grouped:
        time_step = np.diff(np.unique(group['time']))[0]
        simulation_data=group.groupby('Droplet')
        sim = pd.DataFrame()
        data_list = []
        for droplet, data in simulation_data:
            all_zeros = data['Count'].unique()
            if len(all_zeros) == 1 and all_zeros[0] == 0:
                continue
            data = data.drop(columns=['Slice', 'Droplet', 'Unnamed: 0', 'log_Volume', 'Volume', 'Bins_vol', 'Bins_vol_txt', 'Area','InitialOD', 'DW', 'time'])
            data = data.reset_index(drop=True)
            data = data.transpose()
            data_list.append(data)
        sim = pd.concat(data_list, axis=0).reset_index(drop=True)
        sim.to_csv(os.path.join(run_directory, f'{name} experimental simulation data (time step {time_step}).txt'),sep='\t', index=False)
def scanning_GUI():
    root = tk.Tk()
    root.title("Parameter Selection")
    def creating_combinations():
        parameters = run_simulation()
        my_dict_float = {
            key: [float(value) if isinstance(value, str) and value.replace('.', '', 1).isdigit() else value for value in
                  values_list] for key, values_list in parameters.items()}
        values = list(my_dict_float.values())
        non_empty_values = [sublist for sublist in values if sublist]
        for i in range(1, len(non_empty_values)):
            param_name = list(my_dict_float.keys())[i]
            if isinstance(non_empty_values[i][0], float) and len(non_empty_values[i]) == 3:
                if linear_vars[param_name].get():
                    non_empty_values[i] = list(
                        np.linspace(non_empty_values[i][0], non_empty_values[i][1], int(non_empty_values[i][2])))
                elif log_vars[param_name].get():
                    non_empty_values[i][0] = np.log10(non_empty_values[i][0])
                    non_empty_values[i][1] = np.log10(non_empty_values[i][1])
                    non_empty_values[i] = list(
                        np.logspace(non_empty_values[i][0], non_empty_values[i][1], int(non_empty_values[i][2])))
                else:
                    tk.messagebox.showerror("Error",
                                            "Please select either Linear or Logarithmic scale for the parameter.")
                    return
        combinations = list(product(*non_empty_values))
        original_combinations = combinations.copy()  # Store the original combinations
        if fix_waterscape_var.get():
            if len(parameters["Distribution type"]) > 1 or len(parameters["Volume"]) > 1 or len(
                    parameters["Mean"]) > 1 or len(parameters["Standard Deviation"]) > 1:
                tk.messagebox.showerror("Error",
                                        "Only one distribution type, volume, mean, and std can be selected when fixing waterscape.")
                return
            elif fix_bacterial_distribution_var.get() and len(parameters["Number of Bacteria"]) > 1:
                tk.messagebox.showerror("Error",
                                        "Only one value of bacteria can be selected when fixing bacterial distribution.")
                return
            data = main.create_data_by_volume(non_empty_values[2][0], non_empty_values[3][0],non_empty_values[1][0], non_empty_values[0][0])
            if fix_bacterial_distribution_var.get():
                data = main.distribute_bacteria(int(non_empty_values[4][0]), data)
                del non_empty_values[:5]
            else:
                del non_empty_values[:4]
            combinations = list(product(*non_empty_values))
        else:
            data = None
        new_combinations = []
        for combination in combinations:
            if any(param in combination for param in ("logistic", "gompertz")) and isinstance(combination[-1], float):
                combination = combination[:-1]
            new_combinations.append(combination)
        combinations = list(set(new_combinations))
        new_combinations = []
        for combination in original_combinations:
            if any(param in combination for param in ("logistic", "gompertz")) and isinstance(combination[-1], float):
                combination = combination[:-1]
            new_combinations.append(combination)
        original_combinations = list(set(new_combinations))
        if len(combinations[0]) < 9:
            combinations = [(data, *combination) for combination in combinations]
        num_of_replicates = int(num_replicates_entry.get())
        combinations = [combination + (f"replicate {i + 1}",) for i in range(num_of_replicates) for combination in combinations]
        original_combinations = [combination + (f"replicate {i + 1}",) for i in range(num_of_replicates) for combination in original_combinations]
        return original_combinations, combinations, data if data is not None else None

    def custom_combinations(grouped):
        params = run_simulation()
        if all(not value for value in params.values()):
            tk.messagebox.showerror("Error", "Please select at parameters for the simulation.")
            return
        original_combinations, combinations, data = creating_combinations()
        if fix_waterscape_var.get():
            if fix_bacterial_distribution_var.get():
                combinations = [combination[1:] for combination in combinations]
            else:
                combinations = [combination[2:] for combination in combinations]
        else:
            combinations = [combination[5:] for combination in combinations]
        combinations = list(set(combinations))
        new_combinations = []
        for name, group in grouped:
            raw_data = group[['Volume', 'Count', 'log_Volume']].copy()
            raw_data.rename(columns={'Volume': 'droplet size', 'Count': 'num of bacteria', 'log_Volume': 'bin'},
                            inplace=True)
            raw_data = raw_data.reset_index(drop=True)
            raw_data['initial density'] = raw_data['num of bacteria'] / raw_data['droplet size']
            raw_data['Rs'] = np.zeros(len(raw_data))
            valid_density_rows = raw_data['initial density'] > 0
            raw_data.loc[valid_density_rows, 'Rs'] = -5.1 - 0.0357 * np.log2(
            raw_data.loc[valid_density_rows, 'droplet size']) - 0.81 * np.log2(raw_data.loc[valid_density_rows, 'initial density'])
            raw_data['k'] = raw_data['num of bacteria'] * (2 ** raw_data['Rs'])
            raw_data.loc[raw_data['num of bacteria'] > raw_data['k'], 'k'] = raw_data['num of bacteria']
            merged_data = [(raw_data, *combination) for combination in combinations]
            new_combinations.extend(merged_data)
        combinations = new_combinations
        combinations = [(*combination,f'Chip {i}') for i, combination in enumerate(combinations, start=1)]
        num_combinations = len(combinations)  # replace with the actual number of combinations
        original_combinations = [(f'Chip {i}') for i in range(1, num_combinations + 1)]
        fixed_bacterial_distribution(*combinations,*original_combinations)
        scanning_analysis.main()


    def upload_csv():
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            df = pd.read_csv(filename)
            grouped = df.groupby('Slice')
            unique_times = np.unique(df['time'])
            time_steps = np.diff(unique_times)
            if len(time_steps) == 0:
                custom_combinations(grouped)
                return
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            run_directory = os.path.join(os.path.expanduser('~'), 'Desktop', 'raw_data', 'scanning_tool',
                                         f'{timestamp} experimental data')
            os.makedirs(run_directory, exist_ok=True)
            csv_to_raw_data(grouped, run_directory)
            csv_to_simulation_data(df, run_directory)
            tk.messagebox.showinfo("Success", "CSV files saved successfully.")
            scanning_analysis.main()

    def process_combinations():
        original_combinations, combinations, data = creating_combinations()
        if isinstance(data, pd.DataFrame):
            fixed_bacterial_distribution(*combinations,*original_combinations)
        elif isinstance(data, np.ndarray):
            fixed_waterscape(*combinations,*original_combinations)
        else:
            scanning(*combinations,*original_combinations)
        scanning_analysis.main()


    def handle_checkbox_selection(clicked_checkbox, checkboxes, param_type_var):
        distribution_checkboxes = ["lognormal", "truncated normal", "uniform"]
        # If the clicked checkbox is a distribution type checkbox
        if clicked_checkbox.cget("text").lower() in distribution_checkboxes:
            param_type = param_type_var.get()
            # If the mode is "Static", only one checkbox can be selected
            if param_type == "Static":
                if clicked_checkbox.var.get():
                    # Ensure only one checkbox can be selected
                    for checkbox in checkboxes:
                        if checkbox is not clicked_checkbox:
                            checkbox.var.set(False)
            return

        # If the clicked checkbox is not a distribution type checkbox, proceed as before
        param_type = param_type_var.get()
        richards_selected = False

        # Check if the clicked checkbox is "Richards"
        if clicked_checkbox.cget("text").lower() == "richards":
            richards_selected = clicked_checkbox.var.get()

            for param_entry, param_type_var, *other_entries in param_entries:
                param_name = param_entry.cget("text")

                if param_name == "m parameter (for Richards model)":
                    # Enable/disable m parameter text box based on the state of "Richards" checkbox
                    for entry_widget in other_entries:
                        if richards_selected:
                            entry_widget.configure(state='normal')
                        else:
                            entry_widget.configure(state='disabled')

        # If the mode is "Static", only one checkbox can be selected
        if param_type == "Static":
            if clicked_checkbox.var.get():
                # If the clicked checkbox is "Richards", enable m parameter text box
                if clicked_checkbox.cget("text").lower() == "richards":
                    for param_entry, param_type_var, *other_entries in param_entries:
                        param_name = param_entry.cget("text")
                        if param_name == "m parameter (for Richards model)":
                            for entry_widget in other_entries:
                                entry_widget.configure(state='normal')
                elif "distribution" not in clicked_checkbox.cget(
                        "text").lower():  # Check if clicked checkbox is not from distribution type
                    # If the clicked checkbox is not "Richards" or from distribution type, disable m parameter text box
                    for param_entry, param_type_var, *other_entries in param_entries:
                        param_name = param_entry.cget("text")
                        if param_name == "m parameter (for Richards model)":
                            for entry_widget in other_entries:
                                entry_widget.configure(state='disabled')

                # Ensure only one checkbox can be selected
                for checkbox in checkboxes:
                    if checkbox is not clicked_checkbox:
                        checkbox.var.set(False)
        # If the mode is "Dynamic" and "Richards" checkbox is selected, enable the "m parameter" entry widget
        elif param_type == "Dynamic" and richards_selected:
            for param_entry, param_type_var, *other_entries in param_entries:
                param_name = param_entry.cget("text")
                if param_name == "m parameter (for Richards model)":
                    for entry_widget in other_entries:
                        entry_widget.configure(state='normal')

    linear_vars = {}
    log_vars = {}
    # Function to add a parameter entry row
    def add_param_entry_row(row, param_name):
        param_label = ttk.Label(root, text=param_name)
        param_label.grid(row=row, column=0, padx=5, pady=5)

        param_type_var = tk.StringVar(value="Static")
        param_type_vars.append(param_type_var)
        param_type_menu = ttk.OptionMenu(root, param_type_var, "Static", "Static", "Dynamic",
                                         command=lambda _: update_param_entry_widgets(param_type_var, *entry_widgets))
        param_type_menu.grid(row=row, column=1, padx=5, pady=5)
        names = ["start value", "end value", "number of samples"]
        entry_widgets = []
        if param_name in param_options:
            options = param_options[param_name]
            checkboxes = []
            for i, option in enumerate(options):
                var = tk.BooleanVar()
                check_button = ttk.Checkbutton(root, text=option, variable=var)
                check_button.var = var
                check_button.grid(row=row, column=2 + i, padx=5, pady=5, sticky="w")
                check_button.configure(
                    command=lambda cb=check_button: handle_checkbox_selection(cb, checkboxes, param_type_var))
                checkboxes.append(check_button)
                entry_widgets.append(check_button)

        elif param_name == "m parameter (for Richards model)":
            entry_widget = ttk.Entry(root)
            entry_widget.grid(row=row, column=2, padx=5, pady=5)
            entry_widgets.append(entry_widget)
            entry_widget.insert(0, f"{names[0]}")
            entry_widget.configure(state='disabled')
            for i in range(2):  # Add two more entry widgets for dynamic mode
                entry_widget = ttk.Entry(root)
                entry_widget.insert(0, f"{names[i+1]}")  # Add a description inside the text box
                entry_widget.grid(row=row, column=i + 3, padx=5, pady=5)
                entry_widgets.append(entry_widget)
                entry_widget.configure(state='disabled')
            # Disable additional entry widgets initially
            for entry_widget in entry_widgets[1:]:
                entry_widget.grid_remove()
            linear_var = tk.BooleanVar()
            log_var = tk.BooleanVar()
            linear_checkbox = ttk.Checkbutton(root, text="Linear", variable=linear_var,
                                              command=lambda: log_var.set(not linear_var.get()))
            linear_checkbox.configure(state='disabled')
            log_checkbox = ttk.Checkbutton(root, text="Log", variable=log_var,
                                           command=lambda: linear_var.set(not log_var.get()))
            log_checkbox.configure(state='disabled')
            linear_checkbox.grid(row=row, column=5, padx=5, pady=5, sticky="w")
            log_checkbox.grid(row=row, column=6, padx=5, pady=5, sticky="w")
            entry_widgets.extend([linear_checkbox, log_checkbox])
            linear_checkbox.grid_remove()
            log_checkbox.grid_remove()
            linear_vars[param_name] = linear_var
            log_vars[param_name] = log_var
        else:
            entry_widget = ttk.Entry(root)  # Add a description inside the text box
            entry_widget.grid(row=row, column=2, padx=5, pady=5)
            entry_widgets.append(entry_widget)
            entry_widget.insert(0, f"{names[0]}")
            for i in range(2):  # Add two more entry widgets for dynamic mode
                entry_widget = ttk.Entry(root)
                entry_widget.insert(0, f"{names[i+1]}")  # Add a description inside the text box
                entry_widget.grid(row=row, column=i + 3, padx=5, pady=5)
                entry_widgets.append(entry_widget)
            # Disable additional entry widgets initially
            for entry_widget in entry_widgets[1:]:
                entry_widget.grid_remove()
            linear_var = tk.BooleanVar()
            log_var = tk.BooleanVar()
            linear_checkbox = ttk.Checkbutton(root, text="Linear", variable=linear_var,
                                              command=lambda: log_var.set(not linear_var.get()))
            log_checkbox = ttk.Checkbutton(root, text="Log", variable=log_var,
                                           command=lambda: linear_var.set(not log_var.get()))
            linear_checkbox.grid(row=row, column=5, padx=5, pady=5, sticky="w")
            log_checkbox.grid(row=row, column=6, padx=5, pady=5, sticky="w")
            entry_widgets.extend([linear_checkbox, log_checkbox])
            linear_checkbox.grid_remove()
            log_checkbox.grid_remove()
            linear_vars[param_name] = linear_var
            log_vars[param_name] = log_var


        param_entries.append((param_label, param_type_var, *entry_widgets))

    def set_default_values():
        # Set default values for each entry field
        for param_label, param_type_var, *entry_widgets in param_entries:
            param_name = param_label.cget('text')
            if param_name =="Distribution type":
                for checkbox in entry_widgets:
                    if checkbox.cget("text") == "lognormal":
                        checkbox.var.set(True)
                    else:
                        checkbox.var.set(False)
            elif param_name == "Volume":
                entry_widgets[0].delete(0, tk.END)
                entry_widgets[0].insert(0, 100000000)
            elif param_name == "Mean":
                entry_widgets[0].delete(0, tk.END)
                entry_widgets[0].insert(0, 3)
            elif param_name == "Standard Deviation":
                entry_widgets[0].delete(0, tk.END)
                entry_widgets[0].insert(0, 1.5)
            elif param_name == "Number of Bacteria":
                entry_widgets[0].delete(0, tk.END)
                entry_widgets[0].insert(0, 10000)
            elif param_name == "Growth Rate (r)":
                entry_widgets[0].delete(0, tk.END)
                entry_widgets[0].insert(0, 0.5)
            elif param_name == "Time":
                entry_widgets[0].delete(0, tk.END)
                entry_widgets[0].insert(0, 24)
            elif param_name == "Time Steps":
                entry_widgets[0].delete(0, tk.END)
                entry_widgets[0].insert(0, 1)
            elif param_name == "Model":
                for checkbox in entry_widgets:
                    if checkbox.cget("text") == "logistic":
                        checkbox.var.set(True)
                    else:
                        checkbox.var.set(False)
            num_replicates_entry.delete(0, tk.END)
            num_replicates_entry.insert(0, "1")

    def toggle_bacterial_distribution_checkbox():
        if fix_waterscape_var.get():
            fix_bacterial_distribution_checkbox.configure(state='normal')
        else:
            fix_bacterial_distribution_checkbox.configure(state='disabled')
            fix_bacterial_distribution_var.set(False)  # Uncheck the checkbox


    # Add parameter entry rows
    add_param_entry_row(0, "Distribution type")
    add_param_entry_row(1, "Volume")
    add_param_entry_row(2, "Mean")
    add_param_entry_row(3, "Standard Deviation")
    add_param_entry_row(4, "Number of Bacteria")
    add_param_entry_row(5, "Growth Rate (r)")
    add_param_entry_row(6, "Time")
    add_param_entry_row(7, "Time Steps")
    add_param_entry_row(8, "Model")
    add_param_entry_row(9, "m parameter (for Richards model)")
    num_replicates_label = ttk.Label(root, text="Number of replicates:")
    num_replicates_label.grid(row=10, column=0, padx=5, pady=5, sticky="w")
    num_replicates_entry = ttk.Entry(root)
    num_replicates_entry.grid(row=10, column=1, padx=5, pady=5)
    num_replicates_entry.insert(0, 1)  # Set default number of replicates
    fix_waterscape_var = tk.BooleanVar()
    fix_bacterial_distribution_var = tk.BooleanVar()
    fix_waterscape_checkbox = ttk.Checkbutton(root, text="Fix Waterscape", variable=fix_waterscape_var,command=toggle_bacterial_distribution_checkbox)
    fix_bacterial_distribution_checkbox = ttk.Checkbutton(root, text="Fix Bacterial Distribution",variable=fix_bacterial_distribution_var, state='disabled')
    fix_waterscape_checkbox.grid(row=11, column=0, padx=5, pady=5, sticky="w")
    fix_bacterial_distribution_checkbox.grid(row=11, column=1, padx=5, pady=5, sticky="w")

    # Button to run simulation
    run_button = ttk.Button(root, text="Run Simulation", command=process_combinations)
    run_button.grid(row=12, columnspan=5, padx=5, pady=10)
    default_button = ttk.Button(root, text="Set Default Values", command=set_default_values)
    default_button.grid(row=13, columnspan=5, padx=5, pady=10)
    upload_button = ttk.Button(root, text="Upload CSV", command=upload_csv)
    upload_button.grid(row=14, columnspan=5, padx=5, pady=10)
    root.mainloop()

if __name__ == "__main__":
    scanning_GUI()





















