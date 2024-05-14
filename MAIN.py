import streamlit as st
import pandas as pd
import toml
from dotenv import load_dotenv

from src.supabase import SupabaseConnect

import plotly.graph_objs as go
from plotly.subplots import make_subplots
pd.options.plotting.backend = "plotly"
import numpy as np
from src.utils import calculate_trend

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

def analysis_data_by_term(filter_data: pd.DataFrame):

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        term = st.text_input("Từ khóa tìm kiếm: ")

    term = term.lower()
    if "+" in term:
        list_term = term.split("+")
    else:
        list_term = [term]

    with col2:
        negative_term = st.text_input("Từ khóa loại bỏ:")
    
    negative_term = negative_term.lower()

    list_term = [x.strip() for x in list_term]
    list_negative_term = [x.strip() for x in list_negative_term] 

    dict_features = {
        "tổng doanh thu theo tháng": "monthly_revenue_vnd",
        "tổng lượng bán trong tháng": "sold_per_month",
    }

    if term != "":
        filter_data["product_name"] = filter_data["product_name"].str.lower()
        filter_data["sub_category"] = filter_data["sub_category"].str.lower()

        on = st.sidebar.toggle("Phân tích theo Brand")

        filter_data_term = filter_data.copy()
        for x in list_term:
            filter_data_term = filter_data_term.loc[(filter_data_term["product_name"].str.contains(x))]#&(filter_data["sub_category"].str.contains(term))]
        
        if negative_term != "":
            if "+" in negative_term:
                list_negative_term = negative_term.split("+")
            else:
                list_negative_term = [negative_term]

            for x in list_negative_term:
                filter_data_term = filter_data_term.loc[~(filter_data_term["product_name"].str.contains(x, case=False))]

        options = st.radio("Chọn cột phân tích", dict_features.keys(), horizontal=True)

        if on:
            groupdf_sum = filter_data_term.groupby(["brand","time_period"])[dict_features[options]].agg("sum").reset_index()
        else:
            groupdf_sum = filter_data_term.groupby(["time_period"])[dict_features[options]].agg("sum").reset_index()  
        
        groupdf_count = filter_data_term.groupby(["time_period"])[dict_features[options]].agg("count").reset_index()
        # groupdf_count = filter_data_term.groupby(["time_period"])[dict_features[options]].agg("count").reset_index()
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
            for each in groupdf_sum["brand"].unique():
                get_data = groupdf_sum[groupdf_sum["brand"]==each]
                x = list(get_data.time_period.values)
                y = list(get_data[dict_features[options]].values)
                fig.add_trace(go.Scatter(x=x, y=y,
                                            line=dict(shape='linear'),
                                            connectgaps=True,
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
        
        final_df = filter_data_term[selected_columns]
        # seq  = [False]*len(final_df)
        # final_df.insert(0, '', seq)
        show_data(final_df)


def trend_value(nums: list):
    summed_nums = sum(nums)
    multiplied_data = 0
    summed_index = 0 
    squared_index = 0

    for index, num in enumerate(nums):
        index += 1
        multiplied_data += index * num
        summed_index += index
        squared_index += index**2

    numerator = (len(nums) * multiplied_data) - (summed_nums * summed_index)
    denominator = (len(nums) * squared_index) - summed_index**2
    if denominator != 0:
        return numerator/denominator
    else:
        return 0
    

# @st.cache_data
# def cluster_data(filter_data):
#     # tfidf = hero.tfidf(hero.clean(filter_data['product_name'].str.lower()),max_features=500)
#     # filter_data['cluster_label'] = hero.dbscan(tfidf, eps=0.05, min_samples=5)
#     list_sub_cat = filter_data['sub_category'].unique()
#     tc = TextClustering(list_sub_cat)
#     mapping = tc.main_flow()
#     st.write(mapping)
#     st.write(filter_data['sub_category'])
#     filter_data['cluster_label']=  filter_data['sub_category'].apply(lambda x: mapping[x.lower()])

#     return filter_data
@st.cache_data
def get_trend(filter_data):
    list_cluster = filter_data['sub_category'].unique().tolist()
    filter_data["trend"] = [0]*len(filter_data)
    
    for each_cluster  in list_cluster:
        get_data = filter_data.loc[filter_data['sub_category'] == each_cluster]
        indices = get_data.index

        groupdf_sum = get_data.groupby(["time_period"])["sold_per_month"].agg("sum").reset_index()
        if max(groupdf_sum['time_period']) < max(filter_data["time_period"]):
            average = 0
        else:
            average, max_trend, min_trend, overall_trend, extermum_list_x, extermum_list_y = calculate_trend(groupdf_sum["sold_per_month"].values)
        filter_data.loc[indices, "trend"] = average
    return filter_data

if __name__ == '__main__':
    for_developer = False
    set_page_info()
    data = load_data()
    # list_date_range = [x for x in data["time_period"].unique().tolist() if x is not None]
    # list_date_range.sort()
    # start_value, end_value = len(list_date_range)-2,len(list_date_range)-1
    # selection = load_slider('Chọn khoảng thời gian phân tích', 
    #             list_date_range =  list_date_range,
    #             selected=(start_value, end_value), 
    #             )
    # if selection[0] == selection[1]:
    #     filter_data = data.loc[(data["time_period"] == list_date_range[selection[0]]),:] 
    # else:
    #     filter_data = data.loc[(data["time_period"] >= list_date_range[selection[0]]) & (data["time_period"] <= list_date_range[selection[1]]),:] 
    filter_data = data.copy()
    if "df_data" not in st.session_state:
        st.session_state.df_data  = data

    # filter_data = cluster_data(filter_data)
    filter_data = get_trend(filter_data)
    if for_developer == True:
        st.write(filter_data)
    

    analysis_data_by_term(filter_data)
    
    