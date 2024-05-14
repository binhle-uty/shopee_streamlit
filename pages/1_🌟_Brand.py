import streamlit as st
import pandas as pd
import numpy as np
import ast
import toml
import os
import json
from datetime import date
from dotenv import load_dotenv

import matplotlib.pyplot as plt
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from src.utils import convert_money_to_number
from src.supabase import SupabaseConnect

import plotly.express as px
pd.options.plotting.backend = "plotly"

load_dotenv()
config_file = toml.load("config.toml")

PATH = config_file["data"]["shopee"]["local_path"]
SUPABASE_TABLE_NAME = config_file["supabase"]["table"]["table_name"]

choosing_brand = {}
selected_columns = ["brand", "shop_name", "product_name", "price_vnd","category", "sub_category", "sold_per_day","revenue_per_day_vnd","sold_per_month", "monthly_revenue_vnd", "image_link"]
    

def set_page_info():
    st.set_page_config(layout='wide', page_title='Category Analysis', page_icon='./logo/UTY_logo_ORI.png',)
    new_title = '<p style="font-family:sans-serif; color:green; font-size: 42px;">PHÂN TÍCH SẢN PHẨM THEO BRAND</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.text("")

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

def load_data():
    spb = SupabaseConnect()
    table = spb.execute_query(SUPABASE_TABLE_NAME)
    if 'df_data' not in st.session_state:
        st.session_state.df_data = table
    return st.session_state.df_data

def filter_data(df_raw, threshold: int = 10) -> pd.DataFrame:
    """This function to filter the data by the threshold

    Args:
        df_raw (_type_): DataFrame of the raw data.
        threshold (int, optional): threshold to filter. Defaults to 500.

    Returns:
        pd.DataFrame: output DataFrame.
    """
    # df_raw = df_raw.loc[df_raw["brand"]!="-"]
    df_filter = df_raw.loc[df_raw["sold_per_day"].astype(float)>=threshold]
    df_filter = df_filter.sort_values(by=["sold_per_day"],ascending=False)
    return df_filter

def get_selected_data(data, dict_index):
    gb = GridOptionsBuilder.from_dataframe(data)
    
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('multiple', 
                           use_checkbox=True, 
                           groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()
    grid_response = AgGrid(
        data,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=True,
        height=350, 
        width='100%',
        reload_data=True,
        allow_unsafe_jscode=True,
    )

    selected = grid_response['selected_rows'] 
    if selected is not None:
        list_indexx = [list(dict_index[x].values) for x in selected["sub_category"].values]
        list_indexx = [item for sublist in list_indexx for item in sublist]
        st.session_state["selected_rows"].extend(list_indexx)


def get_selected_product(data, dict_index):
    gb = GridOptionsBuilder.from_dataframe(data)
    

    # gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('multiple', 
                           use_checkbox=True, 
                           groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()
    grid_response = AgGrid(
        data,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=True,
        height=350, 
        width='100%',
        reload_data=True,
        allow_unsafe_jscode=True,
    )

    selected = grid_response['selected_rows'] 
    if selected is not None:
        list_indexx = [list(dict_index[x].values) for x in selected["sub_category"].values]
        list_indexx = [item for sublist in list_indexx for item in sublist]
        st.session_state["selected_product"].extend(list_indexx)

def plot_category(selected_df):
    '''function to plot sub_category
    '''
    selected_df = selected_df.reset_index(drop=True)
    selected_df["sub_category"] = selected_df["sub_category"].apply(lambda x: ast.literal_eval(x)[0].lower())

    dict_index = selected_df.groupby(["sub_category"], as_index=False).groups

    col1 , col2, col3 = st.columns(3)
    groups_count = selected_df.groupby("sub_category")["Product"].agg("count").reset_index()[["sub_category", "Product"]]
    groups_count = groups_count.sort_values(by="Product", ascending=False)
    # groups_count = groups_count.head(10)
    # ax, fig = plt.subplots(figsize=(10, 10))
    with col1:
        st.subheader(":pushpin: Top sản phẩm theo số lượng shop", divider='green')
        get_selected_data(groups_count, dict_index)
    
    
    groups_sold_count = selected_df.groupby("sub_category")["sold_per_day"].agg("max").reset_index()[["sub_category", "sold_per_day"]]
    groups_sold_count = groups_sold_count.sort_values(by="sold_per_day", ascending=False)
    # groups_sold_count = groups_sold_count.head(10)
    with col2:
        st.subheader(":bar_chart: Top sản phẩm theo số lượng bán trong ngày", divider='blue')
        get_selected_data(groups_sold_count, dict_index)


    groups_sold_values = selected_df.groupby("sub_category")["revenue_per_day_vnd"].agg("max").reset_index()[["sub_category", "revenue_per_day_vnd"]]
    groups_sold_values = groups_sold_values.sort_values(by="revenue_per_day_vnd", ascending=False)
    # groups_sold_values = groups_sold_values.head(10)
    with col3:
        st.subheader(":moneybag: Top sản phẩm theo doanh thu trong ngày", divider='orange')
        get_selected_data(groups_sold_values, dict_index)

    if len(st.session_state["selected_rows"]) != len(st.session_state["selected_rows_updated"]):
        st.session_state["selected_rows_updated"]= (st.session_state["selected_rows"])
        st.session_state["selected_rows"] =[]

    selected_data = selected_df.iloc[st.session_state["selected_rows_updated"],:].drop_duplicates()
    selected_data = selected_data[selected_columns]
    show_data(selected_data)
    
def plot_pie_brand(data, raw_data):
    data = data.loc[data["brand"]!="-"]
    selected_data = data[selected_columns]
    selected_data = selected_data.sort_values(by="revenue_per_day_vnd", ascending=False)

    show_data(selected_data)
    
def plot_brand_count(data, raw_data):
    data = data.loc[data["brand"]!="-"]
    list_brand = data["brand"].unique().tolist()
    get_raw_data_total = raw_data.loc[raw_data["brand"].isin(list_brand)]
    get_raw_data_category = data.loc[data["brand"].isin(list_brand)]

    fig1, ax = plt.subplots(figsize=(2, 2))
    brand_count_total = get_raw_data_total.groupby("brand")["sold_per_month"].agg("sum").reset_index().sort_values(by="sold_per_month", ascending=False)    
    brand_count_cat = get_raw_data_category.groupby("brand")["sold_per_month"].agg("sum").reset_index().sort_values(by="sold_per_month", ascending=False)    
    
    

    fig1 = brand_count_total.plot(kind="barh", x="sold_per_month", y="brand", 
                            color="brand", title="Số lợng sản phẩm theo Brand",
                            text="sold_per_month", )
    fig1.update(layout_showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

def plot_brand_revenue_all(data, raw_data):
    feature = "monthly_revenue_vnd"
    # data[feature] = data[feature].apply(convert_money_to_number)
    # raw_data[feature] = raw_data[feature].apply(convert_money_to_number)
    data = data.loc[data["brand"]!="-"]
    list_brand = data["brand"].unique().tolist()
    get_raw_data = raw_data.loc[raw_data["brand"].isin(list_brand)]

    fig1, ax = plt.subplots(figsize=(2, 2))
    brand_count = get_raw_data.groupby("brand")[feature].agg("sum").reset_index().sort_values(by=feature, ascending=False)    
    fig1 = brand_count.plot(kind="barh", x=feature, y="brand", 
                            color="brand", title="Brand Revenue", text=[str(x).format('${:,.2f}') for x in brand_count[feature].values.tolist()],
                            )
    fig1.update(layout_showlegend=False)
    st.plotly_chart(fig1,use_container_width=True)

def plot_brand_revenue(data):
    feature = "monthly_revenue_vnd"
    data[feature] = data[feature].apply(convert_money_to_number)
    data = data.loc[data["brand"]!="-"]
    list_brand = data["brand"].unique().tolist()
    get_raw_data = data.loc[data["brand"].isin(list_brand)]

    fig1, ax = plt.subplots(figsize=(2, 2))
    brand_count = get_raw_data.groupby("brand")[feature].agg("sum").reset_index().sort_values(by=feature, ascending=False)    
    
    fig1 = brand_count.plot(kind="barh", x=feature, y="brand", 
                            color="brand", title="Brand Revenue", text=[str(x).format('${:,.2f}') for x in brand_count[feature].values.tolist()],
                            )
    fig1.update(layout_showlegend=False)
    st.plotly_chart(fig1,use_container_width=True)

def plot_save_brand(data, raw_data):
    data = data.loc[data["brand"]!="-"]
    st.subheader("Lưu thông tin Brand", divider='red')

    list_brand = data["brand"].unique().tolist()
    category = data["category"].unique().tolist()
    get_dict_selected = st.session_state["selected_brand"]
    # st.write(get_dict_selected)
    list_selected = [x["chose_brand"] for x in get_dict_selected if x["category"] == category]
    # st.write(list_selected)
    col3, col4 = st.columns([3,1])
    with col3:
        selected_brand = st.multiselect(options=list_brand,
                                        key="brand", 
                                        label="Cuối cùng mình chọn brand",
                                        label_visibility='hidden',
                                        )
    with col4:
        st.write("")
        st.write("")
        st.button(label="Lưu dữ liệu brand", on_click=save_brand)

    #Get raw data
    get_data = raw_data.loc[raw_data["brand"].isin(selected_brand)]
    selected_data = get_data[selected_columns]
    show_data(selected_data)
    
    return selected_brand
    
def save_brand():
    today = date.today()
    fname = f"./output/brand_{today}.json"
    if os.path.exists(fname):
        #read existing file and append new data
        with open(fname,"r") as f:
            loaded = json.load(f)
    loaded = st.session_state["selected_brand"]
    with open(fname,"w", encoding='utf8') as f:
        json.dump(loaded,f, ensure_ascii=False)   

def update_session_brand(selected_cat,select_brand):
    choosing_brand.update({"category":selected_cat, "chose_brand":select_brand})
    
    if not any(d['category'] == selected_cat for d in st.session_state["selected_brand"]):
        st.session_state["selected_brand"].append(choosing_brand)

    to_be_updated_data = {"chose_brand":select_brand}
    item = next(filter(lambda x: x["category"]==selected_cat, st.session_state["selected_brand"]),None)
    if item is not None:
        item.update(to_be_updated_data)

def load_slider(text:str, list_date_range, selected:tuple):
    st.sidebar.subheader(text)
    values = range(len(list_date_range))
    selection = st.sidebar.select_slider(text, 
                                 values, 
                                 value=selected, 
                                 format_func=(lambda x:list_date_range[x]),
                                 label_visibility='hidden')
    
    return selection

def page():
    # st.header("====")
    set_page_info()

    if "df_data" not in st.session_state:
        df_raw = load_data()
        st.session_state.df_data  = df_raw
    else:
        df_raw = st.session_state.df_data


    list_date_range = [x for x in df_raw["time_period"].unique().tolist() if x is not None]
    list_date_range.sort()
    start_value, end_value = len(list_date_range)-2,len(list_date_range)-1
    selection = load_slider('Chọn khoảng thời gian phân tích', 
                list_date_range =  list_date_range,
                selected=(start_value, end_value), 
                )
    if selection[0] == selection[1]:
        filtered_data = df_raw.loc[(df_raw["time_period"] == list_date_range[selection[0]]),:] 
    else:
        filtered_data = df_raw.loc[(df_raw["time_period"] >= list_date_range[selection[0]]) & (df_raw["time_period"] <= list_date_range[selection[1]]),:] 


    if "selected_rows" not in st.session_state:
        st.session_state["selected_brand"] =[]
        st.session_state["selected_rows"] =[]
        st.session_state["selected_rows_updated"] =[]
        st.session_state["selected_product"] =[]
        st.session_state["selected_product_updated"] =[]    

    
    if filtered_data is not None:
        df_filter = filter_data(df_raw=filtered_data)
        category = ["Trái cây sấy khô",
                    "Snack & Bánh kẹo", 
                    "Bánh tráng", 
                    "Sữa",
                    "Nước mắm & nước tương",
                    "Nước mát & nước giải khát",
                    "Hạt khô",
                    "Bún - mì - phở",
                    "Hạt gia vị",
                    "Trà & cà phê",
                    "Dầu ăn",
                    "Các loại khác"]
        
        st.subheader("Chọn Category:")
        selected_cat = st.selectbox("---", options=category,label_visibility="hidden")

        selected_df = df_filter.loc[df_filter["category"].str.lower()==str(selected_cat.lower())]
        # show_data(selected_df)
        
        col1, col2, col3 = st.columns([1,1, 1])
        with col1:
            st.subheader(":paperclip: Sản lượng bán trong tháng của Brand - Category", divider='rainbow')
            plot_brand_count(selected_df, df_filter)
        with col2:
            st.subheader(":moneybag: Tổng doanh thu Brand theo tháng", divider='rainbow')
            plot_brand_revenue_all(selected_df, df_filter)
        with col3:
            st.subheader(":pushpin: Dữ liệu của Brand trong Category", divider='rainbow')
            plot_pie_brand(selected_df, df_filter)
        select_brand =plot_save_brand(selected_df, df_filter)

        update_session_brand(selected_cat,select_brand)
    
if __name__ == '__main__':
    page()

    