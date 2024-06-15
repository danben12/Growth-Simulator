from multiprocessing import Pool
from time import time as T
from scipy.spatial import KDTree
import numpy as np
import psutil
from matplotlib import pyplot as plt
from numba import njit, jit
import pandas as pd
import math
from bokeh.models import Div, HoverTool
from bokeh.layouts import column
from scipy.stats import linregress, lognorm
from bokeh.palettes import Category10
from bokeh.plotting import figure, show
from scipy.stats import truncnorm
from scipy.integrate import solve_ivp
from bokeh.models import CDSView, ColumnDataSource
import cupy as cp




def log_normal_distribution(mean, std, size):
    mean= 10 ** mean
    lognormal_data = lognorm.rvs(std, scale=mean, size=size)
    return lognormal_data

def truncated_normal_distribution(mean, std, size):
    desired_mean = 10 ** mean  # convert the mean to the log scale
    desired_std = 10**std  # convert the standard deviation to the log scale
    lower_bound = 0  # lower bound of the truncated normal distribution
    upper_bound = np.inf  # upper bound of the truncated normal distribution
    a = (lower_bound - desired_mean) / desired_std  # convert the lower bound to the standard deviation units
    b = (upper_bound - desired_mean) / desired_std  # convert the upper bound to the standard deviation units
    trunc = truncnorm(a, b, loc=desired_mean, scale=desired_std)  # create a truncated normal distribution
    samples = trunc.rvs(size)  # generate samples from the truncated normal distribution
    return samples
def uniform_distribution(target_mean, target_std, size):
    target_mean = 10 ** target_mean  # convert the mean to the log scale
    target_std = 10**target_std  # convert the standard deviation to the log scale
    low = target_mean - (target_std * np.sqrt(12) / 2) # lower bound of the uniform distribution
    high = target_mean + (target_std * np.sqrt(12) / 2) # upper bound of the uniform distribution
    low = max(low, 0.1) # make sure the lower bound is not negative
    high = max(high, 0.1) # make sure the upper bound is not negative
    uniform_values = np.random.uniform(low, high, size) # generate uniform data
    return uniform_values



def create_data_by_volume(mean, std, volume, desired_distribution):
    distribution_functions = {'lognormal': log_normal_distribution,
                                'truncated normal': truncated_normal_distribution,
                                'uniform': uniform_distribution}
    distribution_function = distribution_functions[desired_distribution]
    droplet_sizes = distribution_function(mean, std, 1)
    while droplet_sizes.sum() < volume:
            droplet_sizes = np.append(droplet_sizes, distribution_function(mean, std, 10000))
    if droplet_sizes.sum() > volume:
        excess = droplet_sizes.sum() - volume
        for i in range(len(droplet_sizes) - 1, -1, -1):
            if droplet_sizes[i] > excess:
                droplet_sizes[i] -= excess
                break
            else:
                excess -= droplet_sizes[i]
                droplet_sizes = np.delete(droplet_sizes, i)  # use np.delete() instead of remove()
    return droplet_sizes




def distribute_bacteria(pool_size, dataset):
    dataset = np.sort(dataset)
    ratio=dataset/np.sum(dataset) # calculate the ratio of each droplet size
    sum=np.cumsum(ratio)   # calculate the cumulative sum of the ratios
    uni=np.sort(np.random.uniform(0,1,int(pool_size)))
    values_counting, _ = np.histogram(uni, bins=np.concatenate(([0], sum)))
    num_of_bacteria = values_counting
    values_counting = values_counting.cumsum()
    initial_density=num_of_bacteria/dataset.astype(float) # calculate the initial density of each droplet
    Rs=np.zeros(len(dataset)) # create an array to store the Rs value of each droplet
    valid_density_rows = initial_density > 0 # find the rows with valid density
    Rs[valid_density_rows]=-5.1 - 0.0357*np.log2(dataset[valid_density_rows]) - 0.81*np.log2(initial_density[valid_density_rows]) # calculate the Rs value for each droplet
    k=(num_of_bacteria*(2**Rs)) # calculate the k value for each droplet
    # k[(num_of_bacteria > k) & (k == 0)] = num_of_bacteria[(num_of_bacteria > k) & (k == 0)].astype(int)
    bin=np.log10(dataset) # create an array to store the log10 of the droplet size
    df=pd.DataFrame({'droplet size':dataset,'ratio':ratio,'sum':sum,'values counting':values_counting,'num of bacteria':num_of_bacteria,'initial density':initial_density,'Rs':Rs,'k':k,'bin':bin}) # create a dataframe from the arrays
    return df


@njit
def explicit_euler_integration(func, y0, t_eval, *args):
    dt = t_eval[1] - t_eval[0]
    y = np.zeros((len(t_eval), len(y0)))
    y[0] = y0
    replacement_value = args[0]  # Assuming the first argument in *args is the replacement value
    for i in range(1, len(t_eval)):
        y_next = y[i - 1] + dt * func(t_eval[i - 1], y[i - 1], *args)
        if np.any(y_next < 0):
            y_next = np.where(y_next < 0, replacement_value, y_next)  # Replace negative values with replacement_value
            y[i] = y_next
            y[i + 1:] = replacement_value  # Set the rest of the values to the replacement value
            break
        y[i] = y_next
    return y

@njit
def logistic_growth(t,N, K, r):
    return r * (1 - N / K) * N

def logistic_growth_rate(time, N0, K, r,max_step,droplet_size):
    dic={}
    if isinstance(N0,(float,int)):
        N0=[N0]
        K=[K]
        droplet_size=[droplet_size]
    for i in range(len(N0)):
        t_eval = np.arange(0, time + max_step, max_step)
        result=explicit_euler_integration(logistic_growth, [N0[i]], t_eval, K[i], r)
        df = pd.DataFrame({'time': t_eval, 'N': result[:, 0], 'droplet size': droplet_size[i], 'K': K[i]})
        dic[i]=df
    return dic

@njit
def gompertz_growth(t,N, K, r):
    return r * N * np.log(K / N)

def gompertz_growth_rate(time, N0, K, r, max_step,droplet_size):
    dic={}
    if isinstance(N0,(float,int)):
        N0=[N0]
        K=[K]
        droplet_size=[droplet_size]
    for i in range(len(N0)):
        t_eval = np.arange(0, time + max_step, max_step)
        result=explicit_euler_integration(gompertz_growth, [N0[i]], t_eval, K[i], r)
        df = pd.DataFrame({'time': t_eval, 'N': result[:, 0], 'droplet size': droplet_size[i], 'K': K[i]})
        dic[i]=df
    return dic
@njit
def richards_growth(t, N, K, r, m):
        return r * N * (1 - (N / K)**m)

def richards_growth_rate(time, N0, K, r, m, max_step,droplet_size):
    dic={}
    if isinstance(N0,(float,int)):
        N0=[N0]
        K=[K]
        droplet_size=[droplet_size]
    for i in range(len(N0)):
        t_eval = np.arange(0, time + max_step, max_step)
        result = explicit_euler_integration(richards_growth, [N0[i]], t_eval, K[i], r, m)
        df = pd.DataFrame({'time': t_eval, 'N': result[:, 0], 'droplet size': droplet_size[i], 'K': K[i]})
        dic[i]=df
    return dic
def stats_box(model, volume, distribution, mean, std, df, bacteria_pool, time, max_step, r, m):
    if model == 'logistic':  # if the model is logistic
        stats_text = (f"Model: {model}<br>"
            f"Total droplets volume: {volume}<br>"
            f"Distribution type: {distribution}<br>"
            f"log 10 Droplets Mean Size: {mean}<br>"
            f"log 10 Actual Droplets Mean Size: {np.log10(df['droplet size'].mean()):.2f}<br>"
            f"log 10 Droplets Standard Deviation: {std}<br>"
            f"log 10 Actual Droplets Standard Deviation: {np.log10(df['droplet size'].std()):.2f}<br>"
            f"Number of bacteria: {bacteria_pool}<br>"
            f"Time: {time}<br>"
            f"Time Step: {max_step}<br>"
            f"r: {r}<br>"
            f"m: {m}<br>"
        )
    else:
        stats_text = (
            f"Model: {model}<br>"
            f"Total droplets volume: {volume}<br>"
            f"Distribution type: {distribution}<br>"
            f"Droplets Mean Size: {mean}<br>"
            f"Actual Droplets Mean Size: {(df['droplet size'].mean()):.2f}<br>"
            f"Droplets Standard Deviation: {std}<br>"
            f"Actual Droplets Standard Deviation: {(df['droplet size'].std()):.2f}<br>"
            f"Number of bacteria: {bacteria_pool}<br>"
            f"Time: {time}<br>"
            f"Time Step: {max_step}<br>"
            f"r: {r}<br>"
            f"m: {m}<br>"
        )

    stats_div = Div(
        text=stats_text,
        width=500,
        height=400
    )

    stats_div.styles = {
        'text-align': 'left',
        'margin': '10px auto',
        'font-size': '12pt',
        'font-family': 'Arial, sans-serif',
        'color': 'black',
        'background-color': 'lightgray',
        'border': '1px solid black',
        'padding': '20px',
        'box-shadow': '5px 5px 5px 0px lightgray',
        'border-radius': '10px',
        'line-height': '1.5em',
        'font-weight': 'bold',
        'white-space': 'pre-wrap',
        'word-wrap': 'break-word',
        'overflow-wrap': 'break-word',
        'text-overflow': 'ellipsis',
        'hyphens': 'auto'
    }

    return column(stats_div)


def droplet_histogram(df, distribution):
    if distribution == 'lognormal':
        bins = np.logspace(1, np.log10(df['droplet size'].max()), num=16)
    else:
        bins = np.linspace(df['droplet size'].min(), df['droplet size'].max(), num=16)
    # Create a new figure with WebGL backend
    hist = figure(title='Histogram of Droplet Size', x_axis_type='log',
                  x_axis_label='Droplet Size', y_axis_label='Frequency', output_backend="webgl")
    # Calculate histogram data
    hist_data = np.histogram(df['droplet size'], bins=bins)
    hist_data_occupied = np.histogram(df[df['num of bacteria'] > 0]['droplet size'], bins=bins)
    # Create a ColumnDataSource from histogram data
    source = ColumnDataSource(data=dict(
        top=hist_data[0],
        bottom=np.zeros_like(hist_data[0]),
        left=bins[:-1],
        right=bins[1:],
        top_occupied=hist_data_occupied[0]
    ))
    # Create a view on the data source
    view = CDSView(source=source)
    # Plot histogram for all droplet sizes using the view
    hist.quad(top='top', bottom='bottom', left='left', right='right',
              color='gray', alpha=0.5, legend_label='droplet sizes', source=source, view=view)
    # Plot histogram for occupied droplets using the view
    hist.quad(top='top_occupied', bottom='bottom', left='left', right='right',
              color='blue', alpha=0.1, legend_label='occupied droplets', source=source, view=view)
    # Add stats text
    droplet_num = int(df['droplet size'].count())
    occupied_droplets = int(df[df['num of bacteria'] > 0]['droplet size'].count())
    occupancy_rate = occupied_droplets / droplet_num
    stats_text = f"Droplets count: {droplet_num}<br>Occupied Droplets: {occupied_droplets}<br>Occupancy Rate: {occupancy_rate:.2%}"
    stats_div = Div(text=stats_text, width=400, height=100)
    stats_div.styles = {'text-align': 'center', 'margin': '10px auto', 'font-size': '12pt',
                        'font-family': 'Arial, sans-serif', 'color': 'black', 'background-color': 'lightgray',
                        'border': '1px solid black', 'padding': '20px', 'box-shadow': '5px 5px 5px 0px lightgray',
                        'border-radius': '10px', 'line-height': '1.5em', 'font-weight': 'bold',
                        'white-space': 'pre-wrap', 'word-wrap': 'break-word', 'overflow-wrap': 'break-word',
                        'text-overflow': 'ellipsis', 'hyphens': 'auto'}
    # Combine histogram and stats_div into a single plot
    combined_plot = column(hist, stats_div)
    return combined_plot

def N0_Vs_Volume(df):

    source = ColumnDataSource(df)
    view = CDSView(source=source)
    scatter = figure(title='N0 vs. Volume', x_axis_type='log', y_axis_type='log',
                      x_axis_label='Volume', y_axis_label='N0', output_backend="webgl")
    scatter.scatter('droplet size', 'num of bacteria', source=source, view=view, color='gray', alpha=1, legend_label='N0 vs. Volume')
    filtered_df = df[df['num of bacteria'] > 0]
    filtered_df = filtered_df[filtered_df['droplet size'] > 1000]
    filtered_df = filtered_df[filtered_df['droplet size'] > np.mean(filtered_df['droplet size'])]
    x=np.log10(filtered_df['droplet size'])
    y=np.log10(filtered_df['num of bacteria'])
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_values = np.linspace(min(df['droplet size']), max(df['droplet size']), 100)
    y_values = 10 ** (intercept + slope * np.log10(x_values))
    stats_text = f'y = {slope:.2f}x + {intercept:.2f}<br>R² value: {r_value ** 2:.2f}'
    scatter.line(x_values, y_values, color='red', legend_label='Linear Regression')
    stats_div = Div(text=stats_text, width=400, height=100)
    stats_div.styles = {'text-align': 'center', 'margin': '10px auto', 'font-size': '12pt',
                        'font-family': 'Arial, sans-serif', 'color': 'black', 'background-color': 'lightgray',
                        'border': '1px solid black', 'padding': '20px', 'box-shadow': '5px 5px 5px 0px lightgray',
                        'border-radius': '10px', 'line-height': '1.5em', 'font-weight': 'bold',
                        'white-space': 'pre-wrap', 'word-wrap': 'break-word', 'overflow-wrap': 'break-word',
                        'text-overflow': 'ellipsis', 'hyphens': 'auto'}
    combined_plot = column(scatter, stats_div)
    return combined_plot




def K_Vs_Volume(df):
    # Create a ColumnDataSource from df
    source = ColumnDataSource(df)
    # Create a view on the data source
    view = CDSView(source=source)
    scatter = figure(title='k vs. Volume', x_axis_type='log', y_axis_type='log',
                      x_axis_label='Volume', y_axis_label='K', output_backend="webgl")
    scatter.scatter('droplet size', 'k', source=source, view=view, color='gray', alpha=1, legend_label='k vs. Volume')
    filtered_df = df[df['num of bacteria'] > 0]
    filtered_df = filtered_df[filtered_df['droplet size'] > 1000]
    x = np.log10(filtered_df['droplet size'])
    y = np.log10(filtered_df['k'])
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_values = np.linspace(min(df['droplet size']), max(df['droplet size']), 100)
    y_values = 10 ** (intercept + slope * np.log10(x_values))
    stats_text = f'y = {slope:.2f}x + {intercept:.2f}<br>R² value: {r_value ** 2:.2f}'
    scatter.line(x_values, y_values, color='red', legend_label='Linear Regression')
    stats_div = Div(text=stats_text, width=400, height=100)
    stats_div.styles = {'text-align': 'center', 'margin': '10px auto', 'font-size': '12pt',
                        'font-family': 'Arial, sans-serif', 'color': 'black', 'background-color': 'lightgray',
                        'border': '1px solid black', 'padding': '20px', 'box-shadow': '5px 5px 5px 0px lightgray',
                        'border-radius': '10px', 'line-height': '1.5em', 'font-weight': 'bold',
                        'white-space': 'pre-wrap', 'word-wrap': 'break-word', 'overflow-wrap': 'break-word',
                        'text-overflow': 'ellipsis', 'hyphens': 'auto'}
    combined_plot = column(scatter, stats_div)
    return combined_plot
def Initial_Density_Vs_Volume(df):
    # Create a ColumnDataSource from df
    source = ColumnDataSource(df)
    # Create a view on the data source
    view = CDSView(source=source)
    scatter = figure(title='Initial Density vs. Volume', x_axis_type='log', y_axis_type='log',
                      x_axis_label='Volume', y_axis_label='Initial Density', output_backend="webgl")
    scatter.scatter('droplet size', 'initial density', source=source, view=view, color='gray', alpha=1, legend_label='Initial Density vs. Volume')
    rolling_mean = df[df['initial density'] > 0]['initial density'].rolling(window=10).mean()
    scatter.line(df[df['initial density'] > 0]['droplet size'], rolling_mean, color='red', legend_label='Rolling Mean')
    return scatter

def Rs_Vs_Volume(df):
    df = df[df['Rs'] != 0]
    # Create a ColumnDataSource from df
    source = ColumnDataSource(df)
    # Create a view on the data source
    view = CDSView(source=source)
    scatter = figure(title='Rs vs. Volume', x_axis_type='log',
                      x_axis_label='Volume', y_axis_label='Rs', output_backend="webgl")
    scatter.scatter('droplet size', 'Rs', source=source, view=view, color='gray', alpha=1, legend_label='Rs vs. Volume')
    return scatter


def simulation_full_waterscape(time,df,model,r,max_step=0.1,m=None):
    filtered_df = df[df['num of bacteria'] != 0].reset_index(drop=True)
    num_of_bacteria = filtered_df['num of bacteria'].values
    droplet_size = filtered_df['droplet size'].values
    k = filtered_df['k'].values
    model_functions = {
        'logistic': (logistic_growth_rate, 'logistic'),
        'gompertz': (gompertz_growth_rate, 'gompertz'),
        'richards': (richards_growth_rate, 'richards')
    }
    model_function, model_name = model_functions[model]
    if model_name == 'richards':
        dic=model_function(time, num_of_bacteria, k, r, m, max_step, droplet_size)
    else:
        dic=model_function(time, num_of_bacteria, k, r, max_step, droplet_size)
    return dic


def simulate_meta_population(time,df,model,r,m=None,max_step=0.1):
    total_bacteria = df['num of bacteria'].sum()
    total_droplet_size = df['droplet size'].sum()
    initial_density = total_bacteria / total_droplet_size
    Rs=-5.1 - 0.0357*np.log2(total_droplet_size) - 0.81*np.log2(initial_density)
    k=[int(total_bacteria*2**Rs)]
    model_functions = {
        'logistic': (logistic_growth_rate, 'logistic'),
        'gompertz': (gompertz_growth_rate, 'gompertz'),
        'richards': (richards_growth_rate, 'richards')
    }
    model_function, model_name = model_functions[model]
    if model_name == 'richards':
        df2=model_function(time, [total_bacteria], k, r, m, max_step, [total_droplet_size])
    else:
        df2=model_function(time, [total_bacteria], k, r, max_step, [total_droplet_size])
    return df2[0]


def find_time_to_half_capacity(dic):
    T50_vals=[]
    for value in dic.values():
         T50_vals.append(value[value['N'] > (value['N'].max() / 2)][['time','droplet size']].iloc[[0]])
    return T50_vals


def T50_Vs_Volume(dic):
    scatter = figure(title='T50 vs. Volume', x_axis_type='log', x_axis_label='Volume', y_axis_label='T50',
                     output_backend="webgl")
    T50 = find_time_to_half_capacity(dic)
    T50 = pd.concat(T50)
    T50=T50[T50['time'] > 0]
    source = ColumnDataSource(T50)
    view = CDSView(source=source)
    scatter.scatter('droplet size', 'time', source=source, view=view, color='gray', alpha=1,
                    legend_label='T50 vs. Volume')
    return scatter




def Fraction_reched_K(dic):
    combined_df = pd.concat(dic.values(), ignore_index=True)
    combined_df['droplet size'] = np.log10(combined_df['droplet size'])
    combined_df['bottom_bin'] = combined_df['droplet size'].astype(float).apply(math.floor)
    combined_df['top_bin'] = combined_df['droplet size'].astype(float).apply(math.ceil)
    combined_df['reached K'] = np.where(combined_df['N'] >= 0.95 * combined_df['K'], 'Yes', 'No')
    grouped_bins = combined_df.groupby(['bottom_bin', 'top_bin'])
    p = figure(title='Fraction of Population Reached K Over Time for Bin Ranges', x_axis_label='Timestep',y_axis_label='Fraction of Population Reached K', output_backend="webgl")
    color_palette = Category10[10]  # This palette contains 10 different colors
    for i, (bins, group) in enumerate(grouped_bins):
        fraction_reached_K = group.groupby('time').apply(lambda x: (x['reached K'] == 'Yes').sum() / len(x)) * 100
        source = ColumnDataSource(data=dict(x=fraction_reached_K.index, y=fraction_reached_K.values))
        view = CDSView(source=source)
        p.line('x', 'y', source=source, view=view, legend_label=f"Bin Range: {bins[0]}-{bins[1]}", line_width=4,line_color=color_palette[i % len(color_palette)])
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.grid.grid_line_alpha = 0.3
    return p
def Fraction_in_each_bin(dic, time):
    combined_df = pd.concat(dic.values(), ignore_index=True)
    combined_df = combined_df[(combined_df['time'] == 0) | (combined_df['time'] == time)]
    combined_df['bottom_bin'] = np.log10(combined_df['droplet size']).apply(math.floor)
    combined_df['top_bin'] = np.log10(combined_df['droplet size']).apply(math.ceil)
    start_total = combined_df[combined_df['time'] == 0]['N'].sum()
    end_total = combined_df[combined_df['time'] == time]['N'].sum()
    bins = combined_df.groupby(['bottom_bin', 'top_bin', 'time'])['N'].sum()
    bins = bins.unstack().fillna(0)
    bins['start fraction'] = bins[0] / start_total * 100
    bins['end fraction'] = bins[time] / end_total * 100
    p = figure(title='Fraction of Population in Each Bin at Start and End of Simulation', x_axis_label='Bin Range', y_axis_label='Fraction of Population')
    bin_centers = [(start_bin + end_bin) / 2 for start_bin, end_bin in bins.index]
    bin_width = 0.4
    p.vbar(x=[bin_center - bin_width / 2 for bin_center in bin_centers], top=bins['start fraction'], width=bin_width, color='blue', legend_label='Start')
    p.vbar(x=[bin_center + bin_width / 2 for bin_center in bin_centers], top=bins['end fraction'], width=bin_width, color='green', legend_label='End')
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    return p


def check_overlaps(df, positions, radii):
    tree = KDTree(positions)
    for i in range(len(positions)):
        neighbors = tree.query_ball_point(positions[i], r=2 * radii[i])
        for j in neighbors:
            if i != j:  # Exclude self-comparison
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < (radii[i] + radii[j]):
                    return True  # Overlapping droplets found
    return False  # No overlaps found

def optimize_droplet_positions(df, max_attempts=10000):
    df['radius'] = (3 * df['droplet size'] / (4 * np.pi)) ** (1 / 3)
    df['diameter'] = 2 * df['radius']
    df['num of bacteria'] = df['num of bacteria'].astype(int)
    sum_area = df['diameter'].sum()
    radii = df['radius'].values
    num_droplets = len(df)
    for attempt in range(max_attempts):
        x = np.random.uniform(0, sum_area, num_droplets)
        y = np.random.uniform(0, sum_area, num_droplets)
        positions = np.column_stack((x, y))
        if not check_overlaps(df, positions, radii):
            df['x'] = x
            df['y'] = y
            return df  # Valid non-overlapping positions found
        sum_area *= 1.1  # Increase sum_area if overlaps found
    print("Maximum attempts reached. Consider increasing plot dimensions or reducing droplet sizes.")
    return None

def micro_splash_visualization(df):
    p = figure(title="micro splash visualization", output_backend="webgl")
    optimized_df = optimize_droplet_positions(df)
    if optimized_df is None:
        return None  # Failed to find non-overlapping positions within max_attempts
    # Create a ColumnDataSource from optimized_df
    source = ColumnDataSource(optimized_df)
    # Create a view on the data source
    view = CDSView(source=source)
    # Plot droplets using the view
    p.circle(x='x', y='y', radius='radius', source=source, view=view, line_color="blue", fill_color=None, name="droplet")
    # Hover tool
    hover = HoverTool(tooltips=[
        ("Droplet Size", "@{droplet size}{0.00}"),
        ("Number of Bacteria", "@{num of bacteria}")
    ], mode='mouse', renderers=p.select(name="droplet"))
    p.add_tools(hover)
    new_df = optimized_df[['num of bacteria', 'radius', 'x', 'y']].copy()
    new_df['theta'] = new_df['num of bacteria'].apply(lambda x: np.random.uniform(0, 2 * np.pi, x))
    new_df['bac_radius'] = optimized_df.apply(
        lambda row: np.sqrt(np.random.uniform(0, 0.85, int(row['num of bacteria']))) * row['radius'], axis=1)
    new_df['bac_x'] = new_df.apply(lambda row: row['x'] + row['bac_radius'] * np.cos(row['theta']), axis=1)
    new_df['bac_y'] = new_df.apply(lambda row: row['y'] + row['bac_radius'] * np.sin(row['theta']), axis=1)
    combined_x = np.concatenate(new_df['bac_x'].values)
    combined_y = np.concatenate(new_df['bac_y'].values)
    p.circle(combined_x, combined_y, size=3, color="red", alpha=0.5)
    return p
