import time as Time
from bokeh.models import CDSView, ColumnDataSource
import numpy as np
from bokeh.plotting import output_file
from bokeh.models import Div
from bokeh.layouts import column, row
import main
from bokeh.plotting import figure, show
from bokeh.models import ColorBar, LinearColorMapper, LogTicker
from matplotlib import cm
from matplotlib.colors import rgb2hex



line_properties = {'x': 'time', 'y': 'N', 'line_width': 2}
def draw_lines_on_plot(plot, color, source, source_view):
    plot.line(color=color, source=source, **line_properties, view=source_view)
def jet_colormap(num_colors):
    colormap = cm.get_cmap('jet', num_colors)
    colors = [rgb2hex(colormap(i)[:3]) for i in range(colormap.N)]
    return colors

def normalize_data(data):
    max_val = max(data)
    normalized = [x/max_val for x in data]
    return normalized

def graph_panel(df,volume,time,mean,std,distribution,bacteria_pool, r, model,max_step, m=None):
    start = Time.time()
    output_file("simulation_analysis.html", title="Simulation Analysis")
    main_title = "Simulation Analysis"
    stats=main.stats_box(model, volume, distribution, mean, std, df, bacteria_pool, time, max_step, r, m)
    print(f"Stats: {Time.time() - start}")
    hist=main.droplet_histogram(df,distribution)
    print(f"Histogram: {Time.time() - start}")
    N0_Vs_Volume = main.N0_Vs_Volume(df)
    print(f"N0_Vs_Volume: {Time.time() - start}")
    K_Vs_Volume= main.K_Vs_Volume(df)
    print(f"K_Vs_Volume: {Time.time() - start}")
    Initial_Density_Vs_Volume=main.Initial_Density_Vs_Volume(df)
    print(f"Initial_Density_Vs_Volume: {Time.time() - start}")
    Rs_Vs_Volume=main.Rs_Vs_Volume(df)
    print(f"Rs_Vs_Volume: {Time.time() - start}")
    micro_splash_visualization = main.micro_splash_visualization(df)
    print(f"Micro Splash Visualization: {Time.time() - start}")
     # Create a list of colors
    meta_population = main.simulate_meta_population(time,df,model,r,m,max_step)
    print(f"Meta Population: {Time.time() - start}")
    source = ColumnDataSource(meta_population)
    view = CDSView(source=source)
    p1 = figure(title=f'{model.capitalize()} Growth for Different Number of Bacteria (Linear Scale)',
                x_axis_label='Time', y_axis_label='Population', output_backend="webgl")
    p1.line(x='time', y='N', line_width=4, color='black',legend_label='Metapopulation', source=source, view=view)
    p2 = figure(title=f'{model.capitalize()} Normalized Growth for Different Number of Bacteria (Linear Scale)',
                x_axis_label='Time', y_axis_label='Population', output_backend="webgl")
    p3 = figure(title=f'{model.capitalize()} Growth for Different Number of Bacteria (Logarithmic Scale)',
                x_axis_label='Time', y_axis_label='Population', y_axis_type="log", output_backend="webgl")
    p3.line(x='time', y='N', line_width=4, color='black',legend_label='Metapopulation', source=source, view=view)
    p4 = figure(title=f'{model.capitalize()} Normalized Growth for Different Number of Bacteria (Logarithmic Scale)',
                x_axis_label='Time', y_axis_label='Population',y_axis_type="log", output_backend="webgl")
    p1.legend.location = "top_left"  # Adjust location as needed
    p1.legend.click_policy = "hide"  # Click on legend to hide/show lines
    p3.legend.location = "top_left"  # Adjust location as needed
    p3.legend.click_policy = "hide"  # Click on legend to hide/show lines
    dic = main.simulation_full_waterscape(time, df, model, r, max_step, m)
    print(f"Simulation_full_waterscape: {Time.time() - start}")
    T50_Vs_Volume = main.T50_Vs_Volume(dic)
    Fraction_reched_K = main.Fraction_reched_K(dic)
    print(f"Fraction_reched_K: {Time.time() - start}")
    Fraction_in_each_bin = main.Fraction_in_each_bin(dic,time)
    print(f"Fraction_in_each_bin: {Time.time() - start}")
    droplet_sizes = [value['droplet size'].iloc[0] for value in dic.values()]
    max_droplet_size = np.log10(max(droplet_sizes))
    min_droplet_size = np.log10(min(droplet_sizes))
    num_colors = len(droplet_sizes)
    colors = jet_colormap(num_colors)
    color_indices = [list(droplet_sizes).index(droplet_size) for droplet_size in droplet_sizes]
    colors = [colors[color_index] for color_index in color_indices]
    normalized_Ns = [normalize_data(value['N']) for value in dic.values()]
    xs = [value['time'] for value in dic.values()]
    ys = [value['N'] for value in dic.values()]
    normalized_ys = [normalized_N for normalized_N in normalized_Ns]
    source = ColumnDataSource(data={'xs': xs, 'ys': ys, 'colors': colors})
    normalized_source = ColumnDataSource(data={'xs': xs, 'ys': normalized_ys, 'colors': colors})
    p1.multi_line(xs='xs', ys='ys', color='colors', source=source)
    p2.multi_line(xs='xs', ys='ys', color='colors', source=normalized_source)
    p3.multi_line(xs='xs', ys='ys', color='colors', source=source)
    p4.multi_line(xs='xs', ys='ys', color='colors', source=normalized_source)
    mapper = LinearColorMapper(palette=colors, low=min_droplet_size, high=max_droplet_size)
    color_bar = ColorBar(color_mapper=mapper, width=8, location=(0, 0),
                         ticker=LogTicker(desired_num_ticks=10))
    color_bar_plot = figure(title="droplet size Log 10", title_location="right",
                            toolbar_location=None, min_border=0,
                            outline_line_color=None, output_backend="webgl")
    color_bar_plot.add_layout(color_bar, 'right')
    color_bar_plot.title.align = "center"
    color_bar_plot.title.text_font_size = '12pt'
    color_bar_plot.width = 100
    layout1 =row(p1,color_bar_plot)
    layout2=row(p2,color_bar_plot)
    layout3=row(p3,color_bar_plot)
    layout4=row(p4,color_bar_plot)
    print(f"growth curves: {Time.time() - start}")
    plot= [hist, N0_Vs_Volume, K_Vs_Volume, Initial_Density_Vs_Volume, Rs_Vs_Volume, T50_Vs_Volume, Fraction_reched_K, Fraction_in_each_bin, layout1, layout2, layout3, layout4, micro_splash_visualization]
    layout = column(row(stats),row(hist, N0_Vs_Volume), row(K_Vs_Volume, Initial_Density_Vs_Volume), row(Rs_Vs_Volume, layout1),row(layout2,layout3),row(layout4,T50_Vs_Volume),row(Fraction_reched_K,Fraction_in_each_bin),row(micro_splash_visualization))
    # Show the layout
    show(column(Div(text=main_title, styles={'font-size': '20pt', 'text-align': 'center', 'margin-bottom': '20px'}),
                layout))
    print(f"Total time: {Time.time() - start}")
    return dic,plot



