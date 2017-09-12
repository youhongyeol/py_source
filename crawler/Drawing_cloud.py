import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from os import path
from PIL import Image

text_name = '인터랙티비.txt'

def Drawing_cloud(x):  # 추출한 Content 로 WordCloud 를 그립니다.
    d = path.dirname("D:\\portfolio\\")  # 파일의 경로를 설정해주세요.

    # Read the whole text.
    text = open(path.join(d, x), mode="r", encoding='UTF-8').read()
    alice_mask = np.array(Image.open(path.join(d, "cloud.png")))

    stopwords = set(STOPWORDS)
    stopwords.add("said")

    wc = WordCloud(background_color="white",
                   max_words=1000,
                   mask=alice_mask,
                   stopwords=stopwords,
                   font_path='C://Windows//Fonts//BMJUA_ttf.ttf',  # ★Font 를 잘 설정해주셔야합니다!!!★
                   colormap='rainbow_r')

    # generate word cloud
    wc.generate(text)

    # store to file
    wc.to_file(path.join(d, "result.png"))

    # show
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    # plt.figure()
    # plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
    # plt.axis("off")
    plt.show()

Drawing_cloud(text_name)