import pandas as pd
import numpy as np
from underthesea import word_tokenize, pos_tag, sent_tokenize
import string
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import regex
import re


#LOAD EMOJICON
file = open('processing_files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('processing_files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('processing_files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('processing_files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('processing_files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

positive_words_file_path = 'processing_files/positive_VN.txt'
with open(positive_words_file_path, "r", encoding="utf-8") as file:
            positive_words = file.read().splitlines()

negative_words_file_path = 'processing_files/negative_VN.txt'
with open(negative_words_file_path, "r", encoding="utf-8") as file:
            negative_words = file.read().splitlines()

negative_emojis_file_path = 'processing_files/negative_emoji.txt'
with open(negative_emojis_file_path, "r", encoding="utf-8") as file:
            negative_emojis = file.read().splitlines()

positive_emojis_file_path = 'processing_files/positive_emoji.txt'
with open(positive_emojis_file_path, "r", encoding="utf-8") as file:
            positive_emojis = file.read().splitlines()

def replace_phrases(text, replacements):
    """
    Thay thế các cụm từ trong văn bản, ưu tiên cụm dài nhất và bao phủ nhiều nhất.

    :param text: Văn bản đầu vào
    :param replacements: Từ điển các cụm từ cần thay thế
    :return: Văn bản sau khi thay thế
    """
    # Sắp xếp các cụm từ theo độ dài giảm dần để ưu tiên các cụm dài hơn
    sorted_phrases = sorted(replacements.keys(), key=len, reverse=True)

    # Tạo mẫu regex để tìm kiếm các cụm từ
    patterns = []
    for phrase in sorted_phrases:
        # Tạo pattern để khớp chính xác cụm từ, không khớp một phần
        pattern = r'\b{}\b'.format(regex.escape(phrase))
        patterns.append((pattern, replacements[phrase]))

    # Thay thế các cụm từ
    for pattern, replacement in patterns:
        text = regex.sub(pattern, replacement, text)

    return text

def process_text(text, emoji_dict, teen_dict, wrong_lst):
    """
    Xử lý văn bản với các bước chuyển đổi và loại bỏ.

    :param text: Văn bản đầu vào
    :param emoji_dict: Từ điển chuyển đổi emoji
    :param teen_dict: Từ điển teen code và các cụm từ cần thay thế
    :param wrong_lst: Danh sách từ/cụm từ cần loại bỏ
    :return: Văn bản đã xử lý
    """
    # Chuyển về chữ thường và loại bỏ dấu nháy đơn
    document = text.lower()
    document = document.replace("'",'')
    document = regex.sub(r'\.+', ".", document)

    new_sentences = []
    for sentence in sent_tokenize(document):
        # Xử lý emoji
        words = sentence.split()
        emoji_processed_words = [emoji_dict.get(word, word) for word in words]
        sentence = ' '.join(emoji_processed_words)
        # print("Done emoji")

        # Xử lý teen code và các cụm từ thay thế
        sentence = replace_phrases(sentence, teen_dict)
        # print("Done teencode")

        # Loại bỏ các từ sai
        sentence = replace_phrases(sentence, {phrase: '' for phrase in wrong_lst})

        # Giữ lại chỉ các từ tiếng Việt
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern, sentence))
        # print("Done only Vietnamese")

        # # Xử lý từ tiếng anh và thay thế bằng nghĩa tiếng việt
        # sentence = replace_phrases(sentence, english_dict)
        # print("Done english")

        new_sentences.append(sentence.strip())

    # Nối các câu và loại bỏ khoảng trắng thừa
    document = '. '.join(filter(bool, new_sentences)) + '.'
    document = regex.sub(r'\s+', ' ', document).strip()

    return document

# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    # không, chẳng, chả...
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()

# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    # Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    # Ví dụ: "lònggggg" thành "lòng", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)

def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        # lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document

def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def find_words(document, list_of_words):
    document_lower = document.lower()
    word_count = 0
    word_list = []

    for word in list_of_words:
        if word in document_lower:
            print(word)
            word_count += document_lower.count(word)
            word_list.append(word)

    return word_count, word_list

def find_words_list(document, positive_words, negative_words):
    """
    Find and count positive and negative words/phrases with precise overlap handling.

    Args:
    document (str): The input text to search
    positive_words (list): List of positive words/phrases
    negative_words (list): List of negative words/phrases

    Returns:
    tuple: (positive_count, positive_word_list, negative_count, negative_word_list)
    """
    import re

    def get_comprehensive_phrases(words, doc_lower):
        word_matches = []
        for word in words:
            matches = list(re.finditer(re.escape(word.lower()), doc_lower))
            for match in matches:
                word_matches.append({
                    'word': word,
                    'start': match.start(),
                    'end': match.end(),
                    'length': len(word)
                })
        # Sắp xếp theo độ dài (giảm dần) và vị trí (tăng dần)
        word_matches.sort(key=lambda x: (-x['length'], x['start']))
        return word_matches

    # Normalize the document to lowercase for consistent matching
    document_lower = document.lower()

    # Tìm các cụm từ cho positive và negative
    matched_positive = get_comprehensive_phrases(positive_words, document_lower)
    matched_negative = get_comprehensive_phrases(negative_words, document_lower)

    # Hợp nhất danh sách và ưu tiên cụm từ bao phủ lớn nhất
    combined_matches = matched_positive + matched_negative
    combined_matches.sort(key=lambda x: (-x['length'], x['start']))

    selected_words = []
    used_positions = set()
    for match in combined_matches:
        # Kiểm tra xem vị trí có bị trùng không
        if not any(
            (match['start'] < existing_end and match['end'] > existing_start)
            for existing_start, existing_end in used_positions
        ):
            selected_words.append(match)
            used_positions.add((match['start'], match['end']))

    # Phân loại lại positive và negative
    final_positive = [match for match in selected_words if match in matched_positive]
    final_negative = [match for match in selected_words if match in matched_negative]

    # Tính số lượng và danh sách từ
    positive_word_list = [match['word'] for match in final_positive]
    positive_count = len(positive_word_list)

    negative_word_list = [match['word'] for match in final_negative]
    negative_count = len(negative_word_list)

    return positive_count, positive_word_list, negative_count, negative_word_list

def process_comments(row):

    nhan_xet = row['noi_dung_binh_luan']

    # Xử lý nhan_xet qua các hàm
    document1 = process_text(nhan_xet, emoji_dict, teen_dict, wrong_lst)
    document2 = covert_unicode(document1)
    document3 = normalize_repeated_characters(document2)
    document4 = process_postag_thesea(document3)
    document = remove_stopword(document4, stopwords_lst)

    # Tính toán số lượng từ và biểu tượng cảm xúc
    # negative_count, negative_word_list = find_words(document1, negative_words)
    negative_icon, negative_icon_list = find_words(nhan_xet, negative_emojis)
    # positive_count, positive_word_list = find_words(document1, positive_words)
    positive_icon, positive_icon_list = find_words(nhan_xet, positive_emojis)

    positive_count, positive_word_list, negative_count, negative_word_list = find_words_list(document3, positive_words, negative_words)

    # Tính tổng
    total_negative_count = negative_count - negative_icon
    total_positive_count = positive_count - positive_icon

    return pd.Series({
        'comment': document,
        'str_process_text': document1,
        'str_normalize': document3,
        'str_postag': document4,
        'negative_count': total_negative_count,
        'positive_count': total_positive_count,
        'positive_word_list': positive_word_list,
        'negative_word_list': negative_word_list
    })