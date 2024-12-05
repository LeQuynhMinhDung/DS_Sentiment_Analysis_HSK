import pandas as pd

comments = [
    "Tôi không tích sản phẩm này",
    "ok",
    "ổn, bình thường",
    "Không có gì đặc biệt",
    "Rất tốt",
    "Rất thích sản phẩm này"
]

df = pd.DataFrame(comments, columns=["Bình luận"])

df.to_csv('test/new_comments.csv', index=False, header=False, encoding='utf-8')

# Sử dụng phương thức `to_csv` để tạo tệp TXT với tab làm dấu phân cách
df.to_csv('test/new_comments.txt', index=False, header=False, sep='\t', encoding='utf-8')