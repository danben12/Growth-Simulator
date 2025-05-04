import math
import os
import glob
import time
from multiprocessing import Pool
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import dash
from dash import html
from dash import dcc
import webbrowser
from threading import Timer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

start_time= time.time()
def swap_last_with_previous(row):
    last_column_index = len(row) - 1
    if pd.isnull(row.iloc[last_column_index]):
        row.iloc[last_column_index] = row.iloc[last_column_index - 1]
        row.iloc[last_column_index - 1] = None
    return row
def parameter_scanned(filenames):
    parameters_df = pd.DataFrame(filenames)
    parameters_df = parameters_df.apply(swap_last_with_previous, axis=1)
    scanned_parameters = []
    if len(parameters_df.columns) == 10:
        parameters_df.rename(
            columns={0: 'Distribution', 1: 'log 10 Volume', 2: 'Mean', 3: 'std', 4: 'Log 10 Number of bacteria', 5: 'Growth rate',
                     6: 'Time', 7: 'Time steps', 8: 'Model', 9: 'Replicate'}, inplace=True)
    elif len(parameters_df.columns) == 6:
        parameters_df.rename(
            columns={0: 'Log 10 Number of bacteria',1: 'Growth rate',2: 'Time', 3: 'Time steps', 4: 'Model', 5: 'Replicate'}, inplace=True)
    elif len(parameters_df.columns) == 5:
        parameters_df.rename(
            columns={0: 'Growth rate', 1: 'Time', 2: 'Time steps', 3: 'Model',
                     4: 'Replicate'}, inplace=True)
    else:
        parameters_df.rename(
            columns={0: 'Distribution', 1: 'log 10 Volume', 2: 'Mean', 3: 'std', 4: 'Log 10 Number of bacteria', 5: 'Growth rate',
                     6: 'Time', 7: 'Time steps', 8: 'Model', 9: 'm', 10: 'Replicate'}, inplace=True)
    for column in parameters_df.columns:
        unique_values = parameters_df[column].unique()
        if len(unique_values) > 1:
            scanned_parameters.append(column)
    if 'Replicate' in scanned_parameters:
        scanned_parameters.remove('Replicate')
    return parameters_df[scanned_parameters]


def process_simulation_file(file):
    df = pd.read_csv(file,sep="\t")
    filename = os.path.basename(file).replace('.txt', '').replace(' simulation', '').replace('(', '').replace(')',
                                                                                                              '').replace(
        ' ', '').replace('"', '').replace("'", '')
    filename = tuple(filename.split(','))
    return filename, df

def simulations_files(latest_directory):
    files = glob.glob(os.path.join(latest_directory, '*simulation*'))
    with Pool() as pool:
        results = pool.map(process_simulation_file, files)
    simulation_data_dict = dict(results)
    return simulation_data_dict
def process_raw_data_file(file):
    df = pd.read_csv(file,sep="\t")
    filename = os.path.basename(file).replace('.txt', '').replace(' raw_data', '').replace('(', '').replace(')',
                                                                                                              '').replace(
        ' ', '').replace('"', '').replace("'", '')
    filename = tuple(filename.split(','))
    return filename, df
def raw_data_files(latest_directory):
    files = glob.glob(os.path.join(latest_directory, '*raw_data*'))
    with Pool() as pool:
        results = pool.map(process_raw_data_file, files)
    raw_data_dict = dict(results)
    return raw_data_dict
def chip_yield(simulations):
    yield_values = []
    for value in simulations.values():
        sums = pd.DataFrame(value.sum()).transpose()
        yield_values.append(sums.iloc[:,-1].values[0])
    return yield_values
def chip_Rs(simulations):
    Rs_values = []
    for value in simulations.values():
        sums = pd.DataFrame(value.sum()).transpose()
        Rs_values.append(np.log2(sums.iloc[:,-1].values[0]/sums.iloc[:,0].values[0]))
    return Rs_values
def chip_T50(raw_data, simulations):
    T50_values = []
    k_values = []
    Rs_values=chip_Rs(simulations)
    for (value,Rs) in zip(raw_data.values(),Rs_values):
        k = int(value['num of bacteria'].sum() * (2 ** Rs))
        k_values.append(k)
    for (value, k) in zip(simulations.values(), k_values):
        sums = pd.DataFrame(value.sum()).transpose()
        filtered_value = sums[sums > k * 0.5].dropna(axis=1, how='all')
        if not filtered_value.empty:
            T50_values.append(filtered_value.columns[0])
        else:
            T50_values.append(None)  # or any other value that indicates absence of valid data
    T50_values = [float(x) if x is not None else None for x in T50_values]
    return np.array(T50_values)

def calculate_growth_rate(args):
    value = args
    sums = pd.DataFrame(value.sum()).transpose()
    df = pd.DataFrame(columns=['r_squared', 'slope'])
    x = np.array(sums.columns[:5].astype(float)).reshape(-1, 1)
    y = sums.values[0, :5]
    # plt.scatter(x, y)
    # plt.show()
    model = LinearRegression()
    model.fit(x, y)
    r2 = model.score(x, y)
    slope = model.coef_[0]
    df.loc[len(df)] = [r2, np.log10(slope) if slope > 0 else 0]
    return df['slope'].iloc[-1] if not df.empty else None

def chip_growth_rate(simulations):
    with Pool() as pool:
        growth_rate_values = pool.map(calculate_growth_rate,simulations.values())
    return [x for x in growth_rate_values if x is not None]



def create_T50_plot(x,y,i):
    fig1 = go.Figure(data=[go.Scatter(x=x, y=y, mode='markers',name='Chip', marker=dict(size=12))])
    title1 = f'{i} against T50'
    fig1.update_layout(title=title1,height=900,width=1800,xaxis_title=i,yaxis_title='T50')
    return fig1

def create_T50_3D_plot(x,y,z,i,j):
    fig1 = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',name='Chip', marker=dict(size=12))])
    title1 = f'{i} and {j} against T50'
    fig1.update_layout(title=title1,height=900,width=1800,scene=dict(xaxis=dict(title=i),yaxis=dict(title=j),zaxis=dict(title='T50'),aspectmode='cube'))
    return fig1

def create_T50_4D_plot(x, y, z, c, i, j, k):
    fig1 = go.Figure(data=[go.Scatter3d(x=x,y=y,z=z,mode='markers',name='Chip',marker=dict(size=12,color=c,colorscale='Rainbow',colorbar=dict(title=k),opacity=1))])
    title1 = f'{i} and {j} against T50'
    fig1.update_layout(title=title1,height=900,width=1800,scene=dict(xaxis=dict(title=i),yaxis=dict(title=j),zaxis=dict(title='T50'),aspectmode='cube'))
    return fig1

def create_yield_plot(x, y, i):
    fig1 = go.Figure(data=[go.Scatter(x=x, y=y, mode='markers',name='Chip', marker=dict(size=12))])
    title1 = f'{i} against Yield'
    fig1.update_layout(title=title1, height=900, width=1800, xaxis_title=i, yaxis_title='yield')
    return fig1

def create_yield_3D_plot(x, y, z, i, j):
    fig1 = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',name='Chip', marker=dict(size=12))])
    title1 = f'{i} and {j} against Yield'
    fig1.update_layout(title=title1,height=900,width=1800,scene=dict(xaxis=dict(title=i),yaxis=dict(title=j),zaxis=dict(title='Yield'),aspectmode='cube'))
    return fig1

def create_yield_4D_plot(x, y, z, c, i, j, k):
    fig1 = go.Figure(data=[go.Scatter3d(x=x,y=y,z=z,mode='markers',name='Chip',marker=dict(size=12,color=c,colorscale='Rainbow',colorbar=dict(title=k),opacity=1))])
    title1 = f'{i} and {j} against Yield'
    fig1.update_layout(title=title1,height=900,width=1800,scene=dict(xaxis=dict(title=i),yaxis=dict(title=j),zaxis=dict(title='Yield'),aspectmode='cube'))
    return fig1

def create_Rs_plot(x, y, i):
    fig1 = go.Figure(data=[go.Scatter(x=x, y=y, mode='markers',name='Chip', marker=dict(size=12))])
    title1 = f'{i} against Rs'
    fig1.update_layout(title=title1,height=900,width=1800,xaxis_title=i,yaxis_title='Rs')
    return fig1

def create_Rs_3D_plot(x, y, z, i, j):
    fig1 = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',name='Chip', marker=dict(size=12))])
    title1 = f'{i} and {j} against Rs'
    fig1.update_layout(title=title1,height=900,width=1800,scene=dict(xaxis=dict(title=i),yaxis=dict(title=j),zaxis=dict(title='Rs'),aspectmode='cube'))
    return fig1

def create_Rs_4D_plot(x, y, z, c, i, j, k):
    fig1 = go.Figure(data=[go.Scatter3d(x=x,y=y,z=z,mode='markers',name='Chip',marker=dict(size=12,color=c,colorscale='Rainbow',colorbar=dict(title=k),opacity=1))])
    title1 = f'{i} and {j} against Rs'
    fig1.update_layout(title=title1,height=900,width=1800,scene=dict(xaxis=dict(title=i),yaxis=dict(title=j),zaxis=dict(title='Rs'),aspectmode='cube'))
    return fig1

def create_growth_rate_plot(x, y, i):
    fig1 = go.Figure(data=[go.Scatter(x=x, y=y, mode='markers',name='Chip', marker=dict(size=12))])
    title1 = f'{i} against Log10  instantaneous growth rate'
    fig1.update_layout(title=title1,height=900,width=1800,xaxis_title=i,yaxis_title='Log 10 instantaneous growth rate')
    return fig1

def create_growth_rate_3D_plot(x, y, z, i, j):
    fig1 = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',name='Chip', marker=dict(size=12))])
    title1 = f'{i} and {j} against Log 10 instantaneous growth rate'
    fig1.update_layout(title=title1,height=900,width=1800,scene=dict(xaxis=dict(title=i),yaxis=dict(title=j),zaxis=dict(title='Log 10 instantaneous growth rate'),aspectmode='cube'))
    return fig1

def create_growth_rate_4D_plot(x, y, z, c, i, j, k):
    fig1 = go.Figure(data=[go.Scatter3d(x=x,y=y,z=z,mode='markers',name='Chip',marker=dict(size=12,color=c,colorscale='Rainbow',colorbar=dict(title=k),opacity=1))])
    title1 = f'{i} and {j} against Log 10 instantaneous growth rate'
    fig1.update_layout(title=title1,height=900,width=1800,scene=dict(xaxis=dict(title=i),yaxis=dict(title=j),zaxis=dict(title='Log 10 instantaneous growth rate'),aspectmode='cube'))
    return fig1

def index_vs_patches(raw_data,simulations,parameters):
    chip_T50_values = chip_T50(raw_data, simulations)
    chip_yield_values = chip_yield(simulations)
    chip_Rs_values = chip_Rs(simulations)
    chip_growth_rate_values = chip_growth_rate(simulations)
    patches=[]
    Means=parameters['Mean'].values
    initial_Numb_of_bacteria = []
    for (value,mean) in zip(raw_data.values(),Means):
        initial_Numb_of_bacteria.append(np.log10(value['num of bacteria'].sum()))
        patches.append(np.log10(value['droplet size'].sum()/10**mean))
    color_scale = initial_Numb_of_bacteria
    fig1 = go.Figure(data=[go.Scatter(x=patches, y=chip_T50_values, mode='markers',marker=dict(size=12, color=color_scale, colorscale='Rainbow',showscale=True, colorbar=dict(title="Log 10 Number of bacteria")))])
    title1 = f'Number of patches against T50'
    fig1.update_layout(title=title1, height=900, width=1800, xaxis_title='Log 10 Number of patches', yaxis_title='T50')
    fig2 = go.Figure(data=[go.Scatter(x=patches, y=chip_yield_values, mode='markers',marker=dict(size=12, color=color_scale, colorscale='Rainbow',showscale=True, colorbar=dict(title="Log 10 Number of bacteria")))])
    title2 = f'Log 10 Number of patches against Yield'
    fig2.update_layout(title=title2, height=900, width=1800, xaxis_title='Log 10 Number of patches',yaxis_title='Yield')
    fig3 = go.Figure(data=[go.Scatter(x=patches, y=chip_Rs_values, mode='markers',marker=dict(size=12, color=color_scale, colorscale='Rainbow',showscale=True, colorbar=dict(title="Log 10 Number of bacteria")))])
    title3 = f'Number of patches against Rs'
    fig3.update_layout(title=title3, height=900, width=1800, xaxis_title='Log 10 Number of patches', yaxis_title='Rs')
    fig4 = go.Figure(data=[go.Scatter(x=patches, y=chip_growth_rate_values, mode='markers',marker=dict(size=12, color=color_scale, colorscale='Rainbow',showscale=True, colorbar=dict(title="Log 10 Number of bacteria")))])
    title4 = f'Number of patches against instantaneous growth rate'
    fig4.update_layout(title=title4, height=900, width=1800, xaxis_title='Log 10 Number of patches',yaxis_title='instantaneous growth rate')
    return fig2, fig3,fig1, fig4


def index_Vs_heterogeneity(raw_data,simulations):
    chip_T50_values = chip_T50(raw_data, simulations)
    chip_yield_values = chip_yield(simulations)
    chip_Rs_values = chip_Rs(simulations)
    chip_growth_rate_values = chip_growth_rate(simulations)
    std_values = []
    initial_Numb_of_bacteria = []
    for value in raw_data.values():
        initial_Numb_of_bacteria.append(np.log10(value['num of bacteria'].sum()))
        # value['upper bin']=value['bin'].apply(math.ceil)
        # value['lower bin']=value['bin'].apply(math.floor)
        # grouped = value.groupby(['upper bin', 'lower bin']).size().reset_index(name='counts')
        std_values.append(np.log10(value['droplet size'].std()))
    color_scale = initial_Numb_of_bacteria
    fig1 = go.Figure(data=[go.Scatter(x=std_values, y=chip_T50_values, mode='markers',marker=dict(size=12, color=color_scale, colorscale='Rainbow',showscale=True, colorbar=dict(title="Log 10 Number of bacteria")))])
    title1 = f'Heterogeneity against T50'
    fig1.update_layout(title=title1, height=900, width=1800, xaxis_title='Log 10 Heterogeneity', yaxis_title='T50')
    fig2 = go.Figure(data=[go.Scatter(x=std_values, y=chip_yield_values, mode='markers',marker=dict(size=12, color=color_scale, colorscale='Rainbow',showscale=True, colorbar=dict(title="Log 10 Number of bacteria")))])
    title2 = f'Heterogeneity against Log 10 Yield'
    fig2.update_layout(title=title2, height=900, width=1800, xaxis_title='Log 10 Heterogeneity',yaxis_title='Yield')
    fig3 = go.Figure(data=[go.Scatter(x=std_values, y=chip_Rs_values, mode='markers',marker=dict(size=12, color=color_scale, colorscale='Rainbow',showscale=True, colorbar=dict(title="Log 10 Number of bacteria")))])
    title3 = f'Heterogeneity against Rs'
    fig3.update_layout(title=title3, height=900, width=1800, xaxis_title='Log 10 Heterogeneity', yaxis_title='Rs')
    fig4 = go.Figure(data=[go.Scatter(x=std_values, y=chip_growth_rate_values, mode='markers',marker=dict(size=12, color=color_scale, colorscale='Rainbow',showscale=True, colorbar=dict(title="Log 10 Number of bacteria")))])
    title4 = f'Heterogeneity against instantaneous growth rate'
    fig4.update_layout(title=title4, height=900, width=1800, xaxis_title='Log 10 Heterogeneity',yaxis_title='instantaneous growth rate')
    return fig2, fig3,fig1, fig4

def index_Vs_density(raw_data,simulations):
    chip_T50_values = chip_T50(raw_data, simulations)
    chip_yield_values = chip_yield(simulations)
    chip_Rs_values = chip_Rs(simulations)
    chip_growth_rate_values = chip_growth_rate(simulations)
    density_values = []
    for value in raw_data.values():
        density_values.append(value['num of bacteria'].sum()/value['droplet size'].sum())
    fig1 = go.Figure(data=[go.Scatter(x=density_values, y=chip_T50_values, mode='markers',marker=dict(size=12))])
    title1 = f'Density against T50'
    fig1.update_layout(title=title1, height=900, width=1800, xaxis_title='Density', yaxis_title='T50')
    fig2 = go.Figure(data=[go.Scatter(x=density_values, y=chip_yield_values, mode='markers',marker=dict(size=12))])
    title2 = f'Density against Log 10 Yield'
    fig2.update_layout(title=title2, height=900, width=1800, xaxis_title='Density',yaxis_title='Log 10 Yield')
    fig3 = go.Figure(data=[go.Scatter(x=density_values, y=chip_Rs_values, mode='markers',marker=dict(size=12))])
    title3 = f'Density against Rs'
    fig3.update_layout(title=title3, height=900, width=1800, xaxis_title='Density', yaxis_title='Rs')
    fig4 = go.Figure(data=[go.Scatter(x=density_values, y=chip_growth_rate_values, mode='markers',marker=dict(size=12))])
    title4 = f'Density against instantaneous growth rate'
    fig4.update_layout(title=title4, height=900, width=1800, xaxis_title='Density',yaxis_title='instantaneous growth rate')
    return fig1, fig2, fig3, fig4



def run_dash_app(fig_list,latest_directory):
    app = dash.Dash(__name__)
    app.layout = html.Div(children=[
        html.H1(f'Scanning Analysis {latest_directory}'),
        html.Div(children=[
            dcc.Graph(figure=fig, config={'scrollZoom': True}) for fig in fig_list
        ])
    ])
    Timer(1, lambda: webbrowser.open('http://127.0.0.1:8050/')).start()
    Timer(5, lambda: os._exit(0)).start()
    app.run_server(debug=False, use_reloader=False, port=8050)

def check_and_convert_all(df):
    for column in df.columns:
        if df[column].apply(lambda x: str(x).replace(".", "", 1).isdigit()).all():
            df[column] = df[column].astype(float).round(2)
        if column=='Log 10 Number of bacteria':
            df[column]=np.log10(df[column])
        if column=='log 10 Volume':
            df[column]=np.log10(df[column])
    return df
def count_float_columns(df):
    float_columns = df.select_dtypes(include=['float64'])
    return len(float_columns.columns)



def main():
    parent_directory = r'C:\Users\danbe\Desktop\raw_data\scanning_tool'
    directories = [os.path.join(parent_directory, d) for d in os.listdir(parent_directory) if
                   os.path.isdir(os.path.join(parent_directory, d))]
    latest_directory = max(directories, key=os.path.getmtime)
    sim = simulations_files(latest_directory)
    print(f'unpacking the simulations files took {time.time()-start_time} seconds')
    t= time.time()
    raw_data = raw_data_files(latest_directory)
    print(f'unpacking the raw data files took {time.time()-t} seconds')
    t= time.time()
    parameters= parameter_scanned(sim.keys())
    if parameters.empty:
        parameters['Chip'] = [f'Chip {i}' for i in range(1,len(sim)+1)]
    print(f'extracting the scanned parameters took {time.time()-t} seconds')
    t= time.time()
    parameters = check_and_convert_all(parameters)
    print(f'checking and converting all took {time.time()-t} seconds')
    t= time.time()
    float_columns = count_float_columns(parameters)
    print(f'counting float columns took {time.time()-t} seconds')
    non_float_columns = len(parameters.columns) - float_columns
    t= time.time()
    z1 = chip_T50(raw_data, sim)
    print(f'calculating T50 took {time.time()-t} seconds')
    t= time.time()
    z2 = chip_yield(sim)
    print(f'calculating yield took {time.time()-t} seconds')
    t= time.time()
    z3 = chip_Rs(sim)
    print(f'calculating Rs took {time.time()-t} seconds')
    t= time.time()
    z4 = chip_growth_rate(sim)
    print(f'calculating growth rate took {time.time()-t} seconds')
    t= time.time()
    fig_list=[]
    if len(parameters.columns)==1:
            name=parameters.columns[0]
            x=parameters[name].values
            fig_list.append(create_yield_plot(x,z2,name))
            print(f'creating the yield plot took {time.time()-t} seconds')
            t = time.time()
            fig_list.append(create_Rs_plot(x,z3,name))
            print(f'creating the Rs plot took {time.time()-t} seconds')
            t = time.time()
            fig_list.append(create_T50_plot(x,z1,name))
            print(f'creating the T50 plot took {time.time()-t} seconds')
            t = time.time()
            fig_list.append(create_growth_rate_plot(x, z4, name))
            print(f'creating the growth rate plot took {time.time()-t} seconds')
            t = time.time()
    if len(parameters.columns)==2 :
        name1=parameters.columns[0]
        name2=parameters.columns[1]
        x=parameters[name1].values
        y=parameters[name2].values
        fig_list.append(create_yield_3D_plot(x, y, z2, name1, name2))
        print(f'creating the 3D yield plot took {time.time()-t} seconds')
        t = time.time()
        fig_list.append(create_Rs_3D_plot(x,y,z3,name1,name2))
        print(f'creating the 3D Rs plot took {time.time()-t} seconds')
        t = time.time()
        fig_list.append(create_T50_3D_plot(x, y, z1, name1, name2))
        print(f'creating the 3D T50 plot took {time.time()-t} seconds')
        t = time.time()
        fig_list.append(create_growth_rate_3D_plot(x,y,z4,name1,name2))
        print(f'creating the 3D growth rate plot took {time.time()-t} seconds')
        t = time.time()
        if 'Mean' in parameters.columns:
            fig1, fig2, fig3, fig4 = index_vs_patches(raw_data, sim,parameters)
            fig_list.extend([fig1, fig2, fig3, fig4])
            print(f'creating the index vs patches plots took {time.time()-t} seconds')
            t = time.time()
        fig5, fig6, fig7, fig8 = index_Vs_heterogeneity(raw_data, sim)
        fig_list.extend([fig5, fig6, fig7, fig8])
        print(f'creating the index vs heterogeneity plots took {time.time()-t} seconds')
        t = time.time()
    if len(parameters.columns)==3:
        name1=parameters.columns[0]
        name2=parameters.columns[1]
        name3=parameters.columns[2]
        x=parameters[name1].values
        y=parameters[name2].values
        c=parameters[name3].values
        fig_list.append(create_yield_4D_plot(x, y, z2, c, name1, name2, name3))
        print(f'creating the 4D yield plot took {time.time()-t} seconds')
        t = time.time()
        fig_list.append(create_Rs_4D_plot(x, y, z3, c, name1, name2, name3))
        print(f'creating the 4D Rs plot took {time.time()-t} seconds')
        t = time.time()
        fig_list.append(create_T50_4D_plot(x, y, z1,c , name1, name2, name3))
        print(f'creating the 4D T50 plot took {time.time()-t} seconds')
        t = time.time()
        fig_list.append(create_growth_rate_4D_plot(x, y, z4, c, name1, name2, name3))
        print(f'creating the 4D growth rate plot took {time.time()-t} seconds')
        t = time.time()
        if 'Mean' in parameters.columns:
            fig1, fig2, fig3, fig4 = index_vs_patches(raw_data, sim,parameters)
            fig_list.extend([fig1, fig2, fig3, fig4])
            print(f'creating the index vs patches plots took {time.time()-t} seconds')
            t = time.time()
        fig5, fig6, fig7, fig8 = index_Vs_heterogeneity(raw_data, sim)
        fig_list.extend([fig5, fig6, fig7, fig8])
        print(f'creating the index vs heterogeneity plots took {time.time()-t} seconds')
        t = time.time()


    # fig9, fig10, fig11, fig12 = index_Vs_density(raw_data, sim)
    # fig_list.extend([fig9, fig10, fig11, fig12])
    print(f'creating the plots took {time.time()-t} seconds')
    t= time.time()
    run_dash_app(fig_list,latest_directory)
    print(f'running the dash app took {time.time()-t} seconds')
if __name__ == '__main__':
    main()













