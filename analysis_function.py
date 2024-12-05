import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

# Giả sử bạn đã có DataFrame df và các dữ liệu cần thiết
# data = pd.read_csv('your_data.csv') # Đây là ví dụ, bạn cần đọc dữ liệu từ file của bạn

# Hàm phân tích sản phẩm (đã được bạn cung cấp)
def phan_tich_san_pham(df, ma_san_pham):
    """
    Phân tích chi tiết thông tin sản phẩm từ DataFrame
    """
    # Lọc dữ liệu theo mã sản phẩm
    san_pham = df[df['ma_san_pham'] == ma_san_pham]

    if san_pham.empty:
        return {"error": "Không tìm thấy sản phẩm"}

    # Thông tin cơ bản sản phẩm
    ten_san_pham = san_pham['ten_san_pham'].iloc[0]

    # Phân tích điểm số
    diem_trung_binh = san_pham['so_sao'].mean()
    tong_so_luong_danh_gia = len(san_pham)

    # Phân loại sentiment
    sentiment_counts = san_pham['sentiment'].value_counts().to_dict()

    # Hàm xử lý và tạo wordcloud có khung viền
    def tao_wordcloud(text_series):
        # Kết hợp tất cả text
        text = ' '.join(text_series.dropna())

        # Loại bỏ stopwords và ký tự đặc biệt
        text = re.sub(r'[^\w\s]', '', text.lower())

        # Tạo WordCloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            min_font_size=10
        ).generate(text)

        return wordcloud

    # Hàm trích xuất từ khóa chính
    def extract_keywords(text_series):
        # Kết hợp tất cả text
        full_text = ' '.join(text_series.dropna())

        # Loại bỏ stopwords và ký tự đặc biệt
        words = re.findall(r'\w+', full_text.lower())

        # Đếm từ và lọc ra các từ xuất hiện nhiều
        word_counts = Counter(words)
        top_keywords = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        return top_keywords

    # Tạo wordcloud và từ khóa cho từng loại sentiment
    wordclouds = {}
    keywords = {}
    for sentiment in ['positive', 'neutral', 'negative']:
        sentiment_comments = san_pham[san_pham['sentiment'] == sentiment]['comment']
        if not sentiment_comments.empty:
            wordcloud = tao_wordcloud(sentiment_comments)
            wordclouds[sentiment] = wordcloud
            keywords[sentiment] = extract_keywords(sentiment_comments)

    # Thống kê bình luận theo năm
    san_pham['ngay_binh_luan'] = pd.to_datetime(san_pham['ngay_binh_luan'], format='%d/%m/%Y')
    thong_ke_theo_nam = san_pham.groupby(san_pham['ngay_binh_luan'].dt.year).size()

    return {
        "dataframe": san_pham,
        "ma_san_pham": ma_san_pham,
        "ten_san_pham": ten_san_pham,
        "diem_trung_binh": round(diem_trung_binh, 2),
        "tong_so_luong_danh_gia": tong_so_luong_danh_gia,
        "phan_loai_sentiment": sentiment_counts,
        "wordclouds": wordclouds,
        "tu_khoa_chinh": keywords,
        "thong_ke_theo_nam": dict(thong_ke_theo_nam)
    }

def hien_thi_ket_qua(ket_qua):
    # Set the title for the page
    st.write("### Kết Quả Phân Tích Sản Phẩm")

    # Show basic product details
    st.markdown('#### 1. Thông Tin Sản Phẩm')
    st.write(f"**Mã Sản Phẩm:** {ket_qua['ma_san_pham']}")
    st.write(f"**Tên Sản Phẩm:** {ket_qua['ten_san_pham']}")
    st.write(f"**Điểm Đánh Giá Trung Bình:** {ket_qua['diem_trung_binh']}")
    st.write(f"**Tổng Số Lượng Đánh Giá:** {ket_qua['tong_so_luong_danh_gia']}")
    
    st.write("**Một số bình luận:**")
    # st.dataframe(ket_qua['dataframe'][['noi_dung_binh_luan', 'so_sao', 'sentiment']].head(10))
    # Lọc dữ liệu theo từng loại sentiment
    positive_comments = ket_qua['dataframe'][ket_qua['dataframe']['sentiment'] == 'positive'][['noi_dung_binh_luan', 'so_sao', 'sentiment']].head(3)
    negative_comments = ket_qua['dataframe'][ket_qua['dataframe']['sentiment'] == 'negative'][['noi_dung_binh_luan', 'so_sao', 'sentiment']].head(3)
    neutral_comments = ket_qua['dataframe'][ket_qua['dataframe']['sentiment'] == 'neutral'][['noi_dung_binh_luan', 'so_sao', 'sentiment']].head(3)

    # Hiển thị kết quả
    st.write("- Positive (4-5 sao):")
    st.dataframe(positive_comments)

    st.write("- Neutral (3 sao):")
    st.dataframe(neutral_comments)

    st.write("- Negative (0-2 sao):")
    st.dataframe(negative_comments)
    
    # Display sentiment classification
    st.markdown('#### 2. Phân Loại Sentiment:')
    
    # Sentiment statistics bar chart
    st.write('##### 2.1. Thống Kê Số Lượt Đánh Giá Theo Từng Loại Câu Sentiment')
    sentiment_labels = list(ket_qua['phan_loai_sentiment'].keys())
    sentiment_values = list(ket_qua['phan_loai_sentiment'].values())
    # st.bar_chart(dict(zip(sentiment_labels, sentiment_values)))
    # Sắp xếp sentiment_labels và sentiment_values theo thứ tự giảm dần của giá trị
    sorted_data = sorted(zip(sentiment_values, sentiment_labels), reverse=True)
    sorted_values, sorted_labels = zip(*sorted_data)
    # Vẽ biểu đồ dạng cột dọc với Matplotlib
    fig, ax = plt.subplots(figsize=(8, 5))  # Điều chỉnh kích thước biểu đồ
    ax.bar(sorted_labels, sorted_values, color='skyblue')
    ax.set_title('Thống Kê Số Lượng Sentiment', fontsize=14)
    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('Số Lượng', fontsize=12)

    # Hiển thị giá trị trên đầu các cột
    for i, v in enumerate(sorted_values):
        ax.text(i, v + 0.5, f'{v}', ha='center', fontsize=10)

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Show keyword statistics by sentiment
    st.markdown('##### 2.2. Thống Kê Từ Khóa Chính Theo Từng Loại Câu Sentiment')
    for sentiment, count in ket_qua['phan_loai_sentiment'].items():
        st.write(f"**{sentiment.capitalize()}:** {count} đánh giá")
        
        # # Plot keyword statistics for each sentiment
        # keywords = list(ket_qua['tu_khoa_chinh'][sentiment].keys())
        # keyword_counts = list(ket_qua['tu_khoa_chinh'][sentiment].values())
        # keyword_data = dict(zip(keywords, keyword_counts))
        # st.bar_chart(keyword_data, horizontal=True)
        
        # Lấy dữ liệu và sắp xếp theo thứ tự từ cao xuống thấp
        keywords = list(ket_qua['tu_khoa_chinh'][sentiment].keys())
        keyword_counts = list(ket_qua['tu_khoa_chinh'][sentiment].values())
        sorted_data = sorted(zip(keyword_counts, keywords), reverse=True)  # Sắp xếp giảm dần
        sorted_counts, sorted_keywords = zip(*sorted_data)

        # Vẽ biểu đồ dạng cột ngang với Matplotlib
        fig, ax = plt.subplots(figsize=(8, len(sorted_keywords) * 0.5))  # Điều chỉnh kích thước dựa vào số từ khóa
        ax.barh(sorted_keywords, sorted_counts, color='skyblue')
        ax.set_title(f'Thống Kê Từ Khóa Chính ({sentiment.capitalize()})', fontsize=14)
        ax.set_xlabel('Số Lần Xuất Hiện', fontsize=12)
        ax.set_ylabel('Từ Khóa', fontsize=12)
        ax.invert_yaxis()  # Đảo ngược trục y để từ khóa có giá trị cao nhất ở trên cùng

        # Hiển thị giá trị trên cột
        for i, v in enumerate(sorted_counts):
            ax.text(v, i, f'{v}', va='center', ha='left', fontsize=10)

        # Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)
    
    # Display WordClouds
    st.markdown('##### 2.3. WordCloud Theo Theo Từng Loại Câu Sentiment')
    for sentiment, wordcloud in ket_qua['wordclouds'].items():
        # st.write(f'WordCloud {sentiment.capitalize()}')
        st.markdown(f"<h5 style='text-align: center; font-weight: bold; margin-top: 20px;'>WordCloud {sentiment.capitalize()}</h5>", unsafe_allow_html=True)
        st.image(wordcloud.to_array(), use_container_width=True)

    # Yearly statistics bar chart
    st.markdown('#### 3. Thống Kê Số Lượt Đánh Giá Theo Thời Gian (Năm)')
    years = list(ket_qua['thong_ke_theo_nam'].keys())
    year_values = list(ket_qua['thong_ke_theo_nam'].values())
    # st.bar_chart(dict(zip(years, year_values)))

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    bars = plt.bar(years, year_values, color='skyblue')
    plt.title('Thống Kê Bình Luận Theo Năm', fontsize=10)
    plt.xlabel('Năm', fontsize=10)
    plt.ylabel('Số Lượng Bình Luận', fontsize=10)

    # Hiển thị số lượng bình luận trên cột
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # Vị trí x
            height,  # Vị trí y
            f'{height}',  # Văn bản hiển thị
            ha='center', va='bottom', fontsize=8, color='black'  # Định dạng
        )

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(plt)
