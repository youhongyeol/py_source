import urllib.request
from  bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

binary = 'D:\\chromedriver\\chromedriver.exe'
# 웹 브라우저를 크롬을 사용할거라서 크롬 드라이버를 다운받아 위의 위치에 둔다.
# 팬텀 js 로 하면 백그라운드로 실행 할 수 있다.

browser = webdriver.Chrome(binary) # 브라우저를 인스턴스화 한다.

browser.get("http://search.daum.net/search?w=img&nil_search=btn&DA=NTB&enc=utf8&q=")
# 네이버의 이미지 검색 url 을 받아온다.

elem = browser.find_element_by_id("q")
# 네이버의 이미지 검색에 해당하는 input 창의 id 가 nx_query 여서
# 검색창의 해당 HTML 코드를 찾아서 elem 으로 변수를 설정
# find_elements_by_class_name("") # 클래스 이름을 찾을때의 방법

## 검색어 입력
elem.send_keys("bigdata") # elem 이 input 창과 연결이 되어서
elem.submit() # 웹 에서의 submit 은 엔터의 역활을 한다.

## 반복할 횟수
for i in range(1, 2):
    browser.find_element_by_xpath("//body").send_keys(Keys.END)
    # 브라우저 아무데서나 end 키를 누른하고 해서 페이지가 아래로 내려가지 않아서
    # body 를 활성화 해놓고 end 키를 누른다.
    time.sleep(5) # end 로 내려오는 딜레이가 있어 5초의 sleep을 설정

time.sleep(5) # 네트워크 딜레이의 안정성을 위해서 5초의 sleep 을 설정

html = browser.page_source # 크롬 브라우저의 현재 불러온 소스를 가지고 온다.

soup = BeautifulSoup(html, "lxml")
# beautiful soup 를 사용해서 HTML 코드를 검색할 수 있도록 설정

# print(soup)
# print(len(soup))

def fetch_list_url():
    params = []
    imgList = soup.find_all("img", class_="thumb_img")
    # 네이버 이미지 url 이 있는 img 태그의 _img 클래스로 이동해서
    for im in imgList:
        params.append(im["src"]) # params 리스트 변수에 image url 을 append 한다.
    return params

def fetch_detail_url():
    params = fetch_list_url()
    # print(params)
    a = 1
    for i in params:
        print(i)
        # 다운받을 폴더경로 입력
        urllib.request.urlretrieve(i, "d:/DaumImages/" + str(a) + ".jpg")
        a = a + 1

fetch_detail_url()

browser.quit()