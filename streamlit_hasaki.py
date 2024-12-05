import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
from io import StringIO

from build_project_info_function import show_project_info
from analysis_function import phan_tich_san_pham, hien_thi_ket_qua
from new_prediction_function import process_comments

# Custom CSS for Hasaki-themed design
st.set_page_config(
    page_title="Hasaki Sentiment Analysis", 
    page_icon="💖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced design
st.markdown("""
    <style>
    /* Overall page background */
    body {
        background-color: #f0f4f0;
        color: #2f6e51;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #e6f2e8;
        border-right: 2px solid #2f6e51;
    }

    /* Header and title styles */
    .header-title {
        color: #2f6e51;
        font-family: 'Arial', sans-serif;
        text-align: center;
        padding: 20px;
        background-color: #d0e5d3;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    /* Card-like containers */
    .stContainer {
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(47, 110, 81, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #2f6e51;
            
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    /* Button styling */
    .stButton > button {
        background-color: #2f6e51;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }

    /* Dataframe styling */
    .dataframe {
        border: 1px solid #2f6e51;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Sidebar menu styling */
    [data-testid="stSidebar"] .stRadio > div {
        background-color: #d0e5d3;
        padding: 10px;
        border-radius: 10px;
    }

    /* Text and markdown styling */
    .stMarkdown {
        color: #2f6e51;
    }
    
    /* Center images and charts */
    .stImage, .stPlotlyChart, .stPyplot {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Limit height of images */
    .stImage img {
        max-height: 400px;
        width: auto;
        object-fit: contain;
    }
    
    /* Limit height of matplotlib charts */
    .stPyplot > div {
        max-height: 400px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stPyplot img {
        max-height: 400px;
        width: auto;
        object-fit: contain;
    }
            
    </style>
""", unsafe_allow_html=True)


def main():
    # 1. Read data
    final_data = pd.read_csv("data/final_data.csv", encoding='utf-8')

    # 2. Load models 
    # Đọc model
    model_filename = 'model/sentiment_rf_model.pkl'
    with open(model_filename, 'rb') as file:  
        sentiment_model = joblib.load(file)

    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}


    # Tạo các nút tải cho CSV và TXT
    def to_csv(df):
        return df.to_csv(index=False)

    
    #--------------
    # GUI
    # Logo and Title with Hasaki theme
    st.markdown(f'''
    <div class="header-title">
        <h1 style="color: #2f6e51; margin-bottom: 10px;">🌿 Hasaki Sentiment Analysis</h1>
        <p style="color: #45a049; font-size: 0.9em;">Thấu hiểu cảm xúc của khách hàng</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar customization
    # st.sidebar.markdown(''' 
    #     <div style="text-align: center; margin-bottom: 20px;">
    #         <img src="image/Hasaki_Logo.jpg" style="max-width: 100%; max-height: 200px; border-radius: 10px;">
    #     </div>
    # ''', unsafe_allow_html=True)
    
    st.sidebar.markdown("### 📈 Data Science", unsafe_allow_html=True)
    st.sidebar.markdown("""
    **Các thành viên:**
    - Lê Quỳnh Minh Dung
    - Nguyễn Thùy Trang
    
    **Giáo viên hướng dẫn:** Khuất Thùy Phương
    
    **Thời gian:** 12/2024
    """)
    
    # Main menu with improved styling
    # menu = ["Business Objective", "Build Project", "Product Analysis", "New Prediction"]
    menu = ["Mục Tiêu Dự Án", "Xây Dựng Dự Án", "Phân Tích Sản Phẩm", "Phân Tích Dữ Liệu Mới"]
    choice = st.sidebar.radio("Đề mục", menu)
    
    # Existing menu logic remains the same
    if choice == 'Mục Tiêu Dự Án':
        col1, col2, col3 = st.columns([1,2,1])
        with col2:  
            st.markdown('''<div class="stContainer">
                        <h2 style="color: #2f6e51; font-size: 1.75em; ">🌱 Mục Tiêu Dự Án</h2>
                        </div>
                        ''', unsafe_allow_html=True)
            # st.write("### 🌱 Business Objectives")
            st.write("""
            HASAKI.VN là một nhà bán lẻ uy tín chuyên cung cấp các sản phẩm mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên nghiệp với các cửa hàng trên toàn quốc.  
            
            Ứng dụng của chúng tôi sử dụng **phân tích tình cảm (sentiment)** để giúp HASAKI hiểu rõ hơn về phản hồi của khách hàng, từ đó cải thiện chất lượng sản phẩm, nâng cao sự hài lòng của khách hàng và củng cố hình ảnh thương hiệu.  
            """)
            
            st.image("image/Hasaki_Long.jpg")
            
            st.write("""
            ##### 🎯 Mục Tiêu Chính:
            - **Hiểu rõ tình cảm của khách hàng**: Phân tích các đánh giá và xếp hạng sao để nhận diện các phản hồi tích cực, tiêu cực và mang tính xây dựng.  
            - **Cải thiện chất lượng sản phẩm và dịch vụ**: Sử dụng những thông tin thu được để cải thiện các sản phẩm và dịch vụ.  
            - **Tăng cường sự hài lòng và trung thành của khách hàng**: Giải quyết các mối quan tâm của khách hàng để xây dựng niềm tin.  
            """)  
            
            # st.image("image/sentiment.png", caption="Sentiment Analysis for Customer Reviews")
            st.image("image/sentiment.png")
            # st.markdown('</div>', unsafe_allow_html=True)

    elif choice == 'Xây Dựng Dự Án':
        col1, col2, col3 = st.columns([1,2,1])
        with col2:  
            st.markdown('''<div class="stContainer">
                        <h2 style="color: #2f6e51; font-size: 1.75em; ">🔨 Xây Dựng Dự Án</h2>
                        </div>
                        ''', unsafe_allow_html=True)
            # st.write("### Build Project")

            show_project_info(final_data)
        
    elif choice == 'Phân Tích Sản Phẩm':
        col1, col2, col3 = st.columns([1,2,1])
        with col2:  
            st.markdown('''<div class="stContainer">
                        <h2 style="color: #2f6e51; font-size: 1.75em; ">📊 Phân Tích Sản Phẩm</h2>
                        </div>
                        ''', unsafe_allow_html=True)
            # Tạo giao diện Streamlit
            # st.write("### Phân Tích Sản Phẩm")

            # Chọn sản phẩm từ dropdown
            ten_san_pham = st.selectbox("Chọn tên sản phẩm:", final_data['ten_san_pham'].unique())
            ma_san_pham = final_data[final_data["ten_san_pham"] == ten_san_pham]["ma_san_pham"].iloc[0]

            # Phân tích và hiển thị kết quả
            if st.button("Phân Tích"):
                ket_qua = phan_tich_san_pham(final_data, ma_san_pham)
                if "error" in ket_qua:
                    st.error(ket_qua["error"])
                else:
                    hien_thi_ket_qua(ket_qua)


    elif choice == 'Phân Tích Dữ Liệu Mới':
        col1, col2, col3 = st.columns([1,2,1])
        with col2:  
            st.markdown('''<div class="stContainer">
                        <h2 style="color: #2f6e51; font-size: 1.75em; ">📋 Phân Tích Dữ Liệu Mới</h2>
                        </div>
                        ''', unsafe_allow_html=True)
            # Main content area
            st.write("### Nhập dữ liệu mới")
            input_type = st.radio("Chọn phương thức nhập:", options=("Tải lên (Upload)", "Nhập vào (Input)"))
            
            # Xử lý dữ liệu và dự đoán
            processed_data = None
            
            if input_type == "Tải lên (Upload)":
                # Trình tải file
                uploaded_file = st.file_uploader("Chọn tệp CSV hoặc TXT", type=['csv', 'txt'])
                
                if uploaded_file is not None:
                    try:
                        # Đọc tệp dựa trên loại
                        if uploaded_file.name.endswith('.csv'):
                            # df = pd.read_csv(uploaded_file)
                            # df = pd.read_csv(uploaded_file, header=None, names=['noi_dung_binh_luan'])
                            df = pd.read_csv(uploaded_file, header=None)[0].to_frame(name='noi_dung_binh_luan')
                        else:
                            # Với tệp txt, giả định mỗi dòng là một bình luận
                            df = pd.DataFrame({'noi_dung_binh_luan': uploaded_file.getvalue().decode('utf-8').split('\n')})
                        
                        # Loại bỏ các dòng rỗng
                        df = df[df['noi_dung_binh_luan'].notna() & (df['noi_dung_binh_luan'] != '')]
                        
                        st.write("Dữ liệu đã tải lên:")
                        st.dataframe(df.head())
                    except Exception as e:
                        st.error(f"Lỗi khi đọc tệp: {e}")
                        # return
            
            elif input_type == "Nhập vào (Input)":
                # Khu vực nhập văn bản cho nhiều bình luận
                content = st.text_area("Nhập bình luận (mỗi bình luận một dòng):")
                
                if content:
                    # Tách nội dung thành các dòng và tạo DataFrame
                    df = pd.DataFrame({'noi_dung_binh_luan': content.split('\n')})
                    df = df[df['noi_dung_binh_luan'].notna() & (df['noi_dung_binh_luan'] != '')]
            
            if 'df' in locals() and not df.empty:
                # Xử lý các bình luận
                processed_data = df.apply(process_comments, axis=1)
                
                # Kết hợp dữ liệu gốc và dữ liệu đã xử lý
                combined_data = pd.concat([df, processed_data], axis=1)
                
                # Chuẩn bị dữ liệu cho dự đoán
                # Sử dụng cột 'comment', negative_count và positive_count đã được tiền xử lý
                X_pred_data = combined_data[['comment', 'negative_count', 'positive_count']]
                
                # Dự đoán cảm xúc sử dụng toàn bộ pipeline
                y_pred = sentiment_model.predict(X_pred_data)
                
                # Thêm kết quả cảm xúc vào DataFrame
                combined_data['sentiment'] = [sentiment_map[pred] for pred in y_pred]
                
                # Hiển thị kết quả
                st.markdown("### Kết quả Phân tích Cảm xúc")
                st.dataframe(combined_data[['noi_dung_binh_luan', 'sentiment']])

                sentiment_counts = combined_data['sentiment'].value_counts()

                # Sắp xếp sentiment_labels và sentiment_values theo thứ tự giảm dần của giá trị
                sorted_data = sorted(zip(sentiment_counts.values, sentiment_counts.index), reverse=True)
                sorted_values, sorted_labels = zip(*sorted_data)

                # Vẽ biểu đồ dạng cột dọc với Matplotlib
                fig, ax = plt.subplots(figsize=(8, 5))  # Điều chỉnh kích thước biểu đồ
                ax.bar(sorted_labels, sorted_values, color='skyblue')
                ax.set_title('Thống Kê Số Lượng Sentiment', fontsize=10)
                ax.set_xlabel('Sentiment', fontsize=8)
                ax.set_ylabel('Số Lượng', fontsize=8)

                # Hiển thị giá trị trên đầu các cột
                for i, v in enumerate(sorted_values):
                    ax.text(i, v + 0.05, f'{v}', ha='center', fontsize=6)

                # Hiển thị biểu đồ trong Streamlit
                st.write("**Phân phối Sentiment:**") 
                st.pyplot(fig) 

                st.write("**Lưu kết quả:**") 
                # Hiển thị tùy chọn tải dữ liệu
                csv_data = to_csv(combined_data[['noi_dung_binh_luan', 'sentiment']])
                st.download_button(
                    label="Tải về CSV",
                    data=csv_data,
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )

if __name__ == '__main__':
    main()