from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

import re
import numpy as np
import json
from pydantic import BaseModel, Field


def dettach_ko(test_keyword):
    CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    # 중성 리스트. 00 ~ 20
    JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                     'ㅣ']

    # 종성 리스트. 00 ~ 27 + 1(1개 없음)
    JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                     'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28
    split_keyword_list = list(test_keyword)
    # print(split_keyword_list)

    result = list()
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None and keyword not in [*JUNGSUNG_LIST, *CHOSUNG_LIST]:
            r = []
            char_code = ord(keyword) - BASE_CODE
            char1 = int(char_code / CHOSUNG)
            r.append(CHOSUNG_LIST[char1])
            # print('초성 : {}'.format(CHOSUNG_LIST[char1]))
            char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
            r.append(JUNGSUNG_LIST[char2])
            # print('중성 : {}'.format(JUNGSUNG_LIST[char2]))
            char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
            if char3 == 0:
                r.append('#')
            else:
                r.append(JONGSUNG_LIST[char3])
            # print('종성 : {}'.format(JONGSUNG_LIST[char3]))
            result.append(r)
        else:
            result.append([keyword])
    # result
    return result


def dettach_ko_encode(test_keyword):
    re = []
    for i in dettach_ko(test_keyword):
        r = []
        for l in i:
            r.append(float(t2i[l]))
        re.append(pad_sequences(np.array([r], dtype=float), padding='post', maxlen=3)[0].astype(float))
    return np.array(re, dtype=float)


leng = 135
label = {'hate': 1, 'none': 0, 'offensive': 2, 0: 'none', 1: 'hate', 2: 'offensive'}

with open('./CheckHateSpeech/t2i.json', 'r', encoding='utf-8') as f:
    t2i = json.load(f)

with open('./CheckHateSpeech/i2t.json', 'r', encoding='utf-8') as f:
    i2t = json.load(f)

model = load_model('/CheckHateSpeech/CheckHateSpeech.h5')


class Input(BaseModel):
    text: str = Field(
        title='문장을 입력하세요',
        max_length=128
    )
    max_length: int = Field(
        128,
        ge=5,
        le=128
    )
    repetition_penalty: float = Field(
        2.0,
        ge=0.0,
        le=2.0
    )


class Output(BaseModel):
    generated_text: str


def generate_text(input: Input) -> Output:
    x = pad_sequences([dettach_ko_encode(input.text)], padding='post', maxlen=leng, value=[0.0, 0.0, 0.0])
    x = tf.expand_dims(x, axis=3)
    predict = np.argmax(model.predict(x, steps=5)[0])
    return Output(generated_text=label[predict])
