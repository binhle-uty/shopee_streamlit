import streamlit as st
import pandas as pd
import numpy as np
import ast
import toml
import os
import json
from datetime import date, datetime, timedelta
from dotenv import load_dotenv
import time

from src.supabase import SupabaseConnect

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
pd.options.plotting.backend = "plotly"

load_dotenv()
config_file = toml.load("config.toml")

PATH = config_file["data"]["shopee"]["local_path"]
SUPABASE_TABLE_NAME = config_file["supabase"]["table"]["table_name"]
SUPABASE_TABLE_FEATURE = config_file["supabase"]["table"]["table_features"]

choosing_brand = {}
selected_columns = ["time_period","brand", "shop_name", "product_name", "price_vnd","category", "sub_category", "sold_per_day","revenue_per_day_vnd","sold_per_month", "monthly_revenue_vnd", "image_link"]
    

def set_page_info():
    st.set_page_config(layout='wide', page_title='MAIN PAGE', page_icon='./logo/UTY_logo_ORI.png',)
    # st.image("./logo/UTY_LOGO.png", width=400)
    new_title = '<p style="font-family:sans-serif; font-size: 42px;">PHÂN TÍCH SẢN PHẨM THEO TỪ KHÓA</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.text("")
@st.cache_resource
def get_data_from_supabase():
    spb = SupabaseConnect()
    table = spb.execute_query(SUPABASE_TABLE_NAME)
    return table

def modify_data(table):
    # st.info("Loading Completed!")
    table["year"] = table["time_period"].astype(str).apply(lambda x: x.split("-")[0])
    table["month"] = table["time_period"].astype(str).apply(lambda x: x.split("-")[1])
    # t = st.success(f"Completed modifying data!")
    # time.sleep(1)
    # t.empty()
    return table

def preprocess_data(data):
    data["category"] = data["category"].apply(lambda x: "Others" if x is None else x)
    data["sub_category"] = data["sub_category"].apply(lambda x: "Others" if x is None else x)
    # t = st.success(f"Completed preprocessing data!")
    # time.sleep(1)
    # t.empty()
    return data
    
@st.cache_data
def load_data():
    table = get_data_from_supabase()
    table = modify_data(table)
    table = preprocess_data(table)
    # t = st.success(f"Completed loading data!")
    # time.sleep(1)
    # t.empty()
    return table


def load_slider(text:str, list_date_range, selected:tuple):
    st.sidebar.subheader(text)
    values = range(len(list_date_range))
    selection = st.sidebar.select_slider(text, 
                                 values, 
                                 value=selected, 
                                 format_func=(lambda x:list_date_range[x]),
                                 label_visibility='hidden')
    
    return selection

def plot_category(groupdf, column:str = 'monthly_revenue_vnd'):
    fig = go.Figure()
    for each in groupdf["category"].unique():
        get_data = groupdf[groupdf["category"]==each]
        fig.add_trace(go.Scatter(x=get_data.time_period, y=get_data[column].values,
                                name = each,
                                mode = 'markers+lines',
                                line=dict(shape='linear'),
                                connectgaps=True
                                )
                    )
        fig.update_layout(
            autosize=False,
            width=1000,
            height=800,
        )

    st.plotly_chart(fig)

def show_data(data):
    column_configuration = {
        "image_link": st.column_config.ImageColumn("image_link", help="Image Link From Shopee"),
    }
    st.data_editor(
            data,
            column_config=column_configuration,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
        )
    
def handle_term(term):
    if "+" in term:
        list_term = term.split("+")
    if "|" in term:
        list_term = term.split("|")

    

def analysis_data_by_term(filter_data: pd.DataFrame):
    term = st.text_input("Từ khóa tìm kiếm: ")

    dict_features = {
        "tổng doanh thu theo tháng": "monthly_revenue_vnd",
        "tổng lượng bán trong tháng": "sold_per_month",
    }

    if term != "":
        filter_data["product_name"] = filter_data["product_name"].str.lower()
        filter_data["sub_category"] = filter_data["sub_category"].str.lower()

        on = st.sidebar.toggle("Phân tích theo Category")

        filter_data_term = filter_data.loc[(filter_data["product_name"].str.contains(term))]#&(filter_data["sub_category"].str.contains(term))]
        
        options = st.radio("Chọn cột phân tích", dict_features.keys(), horizontal=True)

        if on:
            groupdf_sum = filter_data_term.groupby(["category","time_period"])[dict_features[options]].agg("sum").reset_index()
        else:
            groupdf_sum = filter_data_term.groupby(["time_period"])[dict_features[options]].agg("sum").reset_index()  

        groupdf_count = filter_data_term.groupby(["time_period"])[dict_features[options]].agg("count").reset_index()
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])    

        
        fig.add_trace(go.Bar(x=groupdf_count.time_period, y=groupdf_count[dict_features[options]].values,
                                    text=groupdf_count[dict_features[options]].values,
                                    textposition='auto',
                                    opacity=0.45,
                                    marker=dict(color='#660066'),
                                    name=f'Số lượng sản phẩm',
                                    ),
                        )
        # fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        if on:
            for each in groupdf_sum["category"].unique():
                get_data = groupdf_sum[groupdf_sum["category"]==each]
                fig.add_trace(go.Scatter(x=get_data.time_period, y=get_data[dict_features[options]].values,
                                            line=dict(shape='linear'),
                                            connectgaps=True,
                                            # marker=dict(color='rgb(255, 0, 0)'),
                                            name=f'{each}',
                                            mode='lines+markers',
                                            ),
                                            secondary_y=True,
                                )
        else:
            fig.add_trace(go.Scatter(x=groupdf_sum.time_period, y=groupdf_sum[dict_features[options]].values,
                                            line=dict(shape='linear'),
                                            connectgaps=True,
                                            marker=dict(color='rgb(255, 0, 0)'),
                                            name=f'{options}',
                                            mode='lines+markers',
                                            ),
                                            secondary_y=True,
                                )
        fig.update_layout(
            autosize=False,
            width=1000,
            height=400,
        )
        show_data(groupdf_sum.T)
        st.plotly_chart(fig)
        
        show_data(filter_data_term[selected_columns])

if __name__ == '__main__':
    set_page_info()
    
    data = load_data()
    list_date_range = [x for x in data["time_period"].unique().tolist() if x is not None]
    list_date_range.sort()
    start_value, end_value = len(list_date_range)-2,len(list_date_range)-1
    selection = load_slider('Chọn khoảng thời gian phân tích', 
                list_date_range =  list_date_range,
                selected=(start_value, end_value), 
                )
    if selection[0] == selection[1]:
        filter_data = data.loc[(data["time_period"] == list_date_range[selection[0]]),:] 
    else:
        filter_data = data.loc[(data["time_period"] >= list_date_range[selection[0]]) & (data["time_period"] <= list_date_range[selection[1]]),:] 
    if "df_data" not in st.session_state:
        st.session_state.df_data  = data

    analysis_data_by_term(filter_data)
    
    