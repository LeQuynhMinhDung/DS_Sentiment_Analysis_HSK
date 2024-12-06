import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def show_project_info(final_data):

    # Dữ liệu và Xử lý Bình luận
    st.write("##### 1. Dữ liệu")
    st.write("Dưới đây là một phần dữ liệu đầu vào:")
    st.dataframe(final_data[['noi_dung_binh_luan','so_sao']].tail(5))

    st.write("##### 2. Xử lý Bình luận")

    # Mô tả các bước xử lý bình luận
    st.write("**Các bước xử lý bình luận:**")
    st.write("- **Chuyển đổi emoji**: Xử lý các emoji trong bình luận để chuẩn hóa văn bản.")
    st.write("- **Xử lý teencode**: Chuyển các từ viết tắt hoặc lóng thành từ chuẩn.")
    st.write("- **Sửa từ viết sai**: Sửa lỗi chính tả để làm sạch dữ liệu.")
    st.write("- **Dịch tiếng Anh sang tiếng Việt**: Chuyển các bình luận tiếng Anh sang tiếng Việt nếu cần.")
    st.write("- **Xử lý stopwords**: Loại bỏ các từ không mang nhiều ý nghĩa trong việc phân tích.")
    st.write("- **Xử lý POS tagging**: Tách các từ quan trọng trong câu bằng cách phân tích ngữ pháp.")

    # Hiển thị ví dụ về bình luận đã được xử lý
    st.write("Ví dụ về các bình luận sau khi xử lý:")
    st.dataframe(final_data[['noi_dung_binh_luan', 'comment']].tail(5))

    # Cột Đếm Từ Tích Cực và Tiêu Cực
    st.write("##### 3. Thêm 2 cột đếm từ tích cực và tiêu cực:")
    st.dataframe(final_data[['noi_dung_binh_luan', 'negative_count', 'positive_count']].tail(5))

    st.write("Thông tin về thêm 2 cột 'positive_count' và 'negative_count':")
    st.write("- **Đếm từ tích cực và tiêu cực**: Sử dụng danh sách các từ tích cực và tiêu cực để đếm số từ xuất hiện trong bình luận.")
    st.write("- **Xử lý cụm từ**: Đối với những cụm từ chứa cả từ tích cực và tiêu cực như 'không tốt lắm', ta chỉ đếm 1 lần từ tiêu cực duy nhất thay vì đếm riêng biệt các từ như 'tốt' và 'không tốt'.")
    st.write("- **Xử lý nhiễu**: Trong trường hợp khách hàng đưa ra bình luận tiêu cực nhưng số sao lại tích cực, ta sẽ so sánh giữa 'positive_count' và 'negative_count'. Nếu số lượng từ tích cực vượt trội hơn, ta sẽ loại bỏ những phần bình luận này đi. Xử lý trường hợp ngược lại cũng tương tự.")

    #  Xử lý Nhãn
    st.write("##### 4. Xử lý Nhãn:")
    st.write("- **Nhãn 0 (negative)**: Các bình luận có số sao từ 0-2 sẽ được gán nhãn là negative.")
    st.write("- **Nhãn 1 (neutral)**: Các bình luận có số sao bằng 3 sẽ được gán nhãn là neutral.")
    st.write("- **Nhãn 2 (positive)**: Các bình luận có số sao từ 4-5 sẽ được gán nhãn là positive.")

    # Hiển thị thông tin về các nhãn
    st.dataframe(final_data[['so_sao', 'sentiment', 'label']].tail(5))

    # Biểu đồ so sánh số lượng từ tích cực và tiêu cực
    st.write("##### 5. Biểu đồ so sánh số lượng từ tích cực và tiêu cực trong các nhãn sentiment:")

    # Tính toán trung bình số từ tích cực và tiêu cực theo từng nhãn sentiment
    grouped_data = final_data.groupby('sentiment')[['negative_count', 'positive_count']].mean().reset_index()

    # Vẽ biểu đồ
    x_labels = grouped_data['sentiment']
    x = np.arange(len(x_labels))  # Vị trí các label trên trục x
    width = 0.35  # Độ rộng của các cột

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, grouped_data['negative_count'], width, label='Từ Tiêu Cực', color='red')
    bars2 = ax.bar(x + width/2, grouped_data['positive_count'], width, label='Từ Tích Cực', color='blue')

    # Thêm nhãn và tiêu đề
    ax.set_xlabel('Loại câu')
    ax.set_ylabel('Số lượng')
    ax.set_title('So sánh số lượng từ Negative và Positive trung bình trong 1 câu theo nhãn')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    # Thêm giá trị trên các cột
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # Hiển thị biểu đồ
    st.pyplot(fig)

    # Thêm bình luận về biểu đồ
    st.write("""
    Số lượng từ Negative và Positive được phân bổ khá hợp lý với các bình luận có nhãn sau:
    - **Nhãn 0 (negative)** có số từ negative lớn hơn số từ positive.
    - **Nhãn 1 (neutral)** có số lượng từ positive lớn hơn negative ở mức độ tương đối.
    - **Nhãn 2 (positive)** có số lượng từ positive vượt trội rất nhiều so với negative trong 1 bình luận.
    """)

    st.markdown(f"<h5 style='text-align: center; font-weight: bold; margin-top: 20px;'>WordCloud Positive</h5>", unsafe_allow_html=True)
    st.image("image/building_model_result/WordCloud_Positive.png")
    st.markdown(f"<h5 style='text-align: center; font-weight: bold; margin-top: 20px;'>WordCloud Neutral</h5>", unsafe_allow_html=True)
    st.image("image/building_model_result/WordCloud_Neutral.png")
    st.markdown(f"<h5 style='text-align: center; font-weight: bold; margin-top: 20px;'>WordCloud Negative</h5>", unsafe_allow_html=True)
    st.image("image/building_model_result/WordCloud_Negative.png")

    # # Hiển thị các sentiment có sẵn trong DataFrame
    # sentiments = final_data['sentiment'].unique()

    # # Tạo một WordCloud cho mỗi loại sentiment
    # for sentiment in sentiments:
    #     # Lọc DataFrame theo sentiment đã chọn
    #     sentiment_data = final_data[final_data['sentiment'] == sentiment]
        
    #     # Tạo chuỗi văn bản từ cột 'comment'
    #     text = " ".join(comment for comment in sentiment_data['comment'])
        
    #     # Tạo WordCloud
    #     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
    #     # Vẽ WordCloud
    #     # st.subheader(f"WordCloud for {sentiment.capitalize()} Sentiment")
    #     st.markdown(f"<h5 style='text-align: center; font-weight: bold; margin-top: 20px;'>WordCloud {sentiment.capitalize()}</h5>", unsafe_allow_html=True)
    #     plt.figure(figsize=(10, 5))
    #     plt.imshow(wordcloud, interpolation='bilinear')
    #     plt.axis('off')
    #     st.pyplot(plt)

    # Thống kê số lượng các nhãn sentiment
    st.write("##### 6. Thống kê số lượng sentiment")
    value_counts = final_data['sentiment'].value_counts()

    # Biểu đồ thống kê
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = value_counts.plot(kind='bar', color='skyblue', ax=ax)

    ax.set_xlabel('Loại câu')
    ax.set_ylabel('Số lần xuất hiện')
    ax.set_title('Số lần xuất hiện của các loại câu trong sentiment')
    ax.set_xticklabels(value_counts.index, rotation=0)

    # Hiển thị số trên các cột
    for rect in ax.patches:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')

    # Hiển thị biểu đồ
    st.pyplot(fig)

    st.write("""
    Dữ liệu có sự mất cân bằng lớn, với số lượng câu thuộc loại "positive" chiếm tới 90% tổng lượng dữ liệu. Sau khi cân nhắc, phương pháp **Class Weights** đã được chọn để xử lý imbalance.
    """)

    st.write("##### 7. Data cuối cùng trước khi đưa vào model")
    st.dataframe(final_data[['comment', 'negative_count', 'positive_count', 'label']].tail(5))

    st.write("##### 8. Xây dựng model")

    st.write("""
    Sau khi thử nghiệm với nhiều loại mô hình khác nhau, **Random Forest Classifier** đã được chọn vì tính hiệu quả và độ chính xác cao, đồng thời áp dụng cả phương pháp **Class Weights** để xử lý imbalance.
    """)

    # Kết quả Mô hình
    st.write("##### 9. Kết quả Mô hình Sentiment Analysis")

    # Hiển thị hình ảnh của Classification Report và Confusion Matrix
    st.write("###### 9.1. Classification Report: ")
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.image("image/building_model_result/Classification_Report_Random_Forest.png")

    st.write("###### 9.2. Confusion Matrix:")
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.image("image/building_model_result/Confusion_Matrix_for_Random_Forest_Classifier.png")

    st.write("Bài toán Sentiment Analysis sử dụng thuật toán Random Forest Classifier đạt kết quả khá tốt với độ chính xác lên đến 98.45%.")
