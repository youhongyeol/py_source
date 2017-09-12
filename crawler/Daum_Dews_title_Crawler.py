import urllib.request
import os
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from os import path
from PIL import Image

class Daum_News_Crawler:
    def __init__(self, keyword, cnt):
        self.keyword = keyword
        self.cnt = cnt + 1

    def get_save_path(self): # txt 파일의 저장경로를 설정합니다.
        save_path = "D:\\portfolio\\" + self.keyword + ".txt" # 파일의 경로를 설정해주세요.
        save_path = save_path.replace("\\", "/")
        if not os.path.isdir(os.path.split(save_path)[0]):
            os.mkdir(os.path.split(save_path)[0])
        return save_path

    def Daum_News(self): # 다음 뉴스의 URL 을 크롤링합니다.
        f = open(self.get_save_path(), 'w', encoding="utf-8")
        for i in range(1, self.cnt):
            binary = 'D:\chromedriver/chromedriver.exe'
            browser = webdriver.Chrome(binary)
            url = 'http://search.daum.net/search?nil_suggest=btn&w=news&DA=PGD&cluster=y&q='+ self.keyword +'&p='+str(i)
            browser.get(url)
            html = browser.page_source
            soup = BeautifulSoup(html, "html.parser")
            for j in range(30):
                try:
                    soup1 = soup.find_all('div', class_='wrap_tit mg_tit')[j]
                    soup2 = soup1.get_text()
                    f.write(soup2)
                except Exception:
                    pass
            browser.quit()
        f.close()

    def Drawing_cloud(self): # 추출한 Content 로 WordCloud 를 그립니다.
        d = path.dirname("D:\\portfolio\\") # 파일의 경로를 설정해주세요.

        # Read the whole text.
        text = open(path.join(d, self.keyword + '.txt'), mode="r", encoding='UTF-8').read()
        alice_mask = np.array(Image.open(path.join(d, "cloud.png")))

        stopwords = set(STOPWORDS)
        stopwords.add("said")

        wc = WordCloud(background_color="white",
                       max_words=1000,
                       mask=alice_mask,
                       stopwords=stopwords,
                       font_path='C://Windows//Fonts//BMJUA_ttf.ttf', # ★Font 를 잘 설정해주셔야합니다!!!★
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

if __name__ == '__main__':
    Daum = Daum_News_Crawler('AI', 30) # 검색어, 페이지 수 를 입력하세요!!
    Daum.Daum_News()
    Daum.Drawing_cloud()