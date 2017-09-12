import urllib.request # 웹브라우저에서 HTML 문서를 얻어오기위해 통신하기 위한 모듈
from bs4 import BeautifulSoup # HTML 문서 검색 모듈
import os

def get_save_path():
    save_path = input("Enter the file name and file location : " )
    save_path = save_path.replace("\\", "/")
    if not os.path.isdir(os.path.split(save_path)[0]):
        os.mkdir(os.path.split(save_path)[0]) # 폴더가 없으면 만드는 작업
    return save_path

def fetch_list_url():
    params = []
    for i in range(1,10):
        list_url = "http://search.joins.com/JoongangNews?page="+str(i)+"&Keyword=%EB%AC%B8%EC%9E%AC%EC%9D%B8&SortType=New&SearchCategoryType=JoongangNews"
        url = urllib.request.Request(list_url)
        # url 요청에 따른 http 통신 헤더값을 얻어낸다
        res = urllib.request.urlopen(url).read().decode("utf-8")
        # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을
        # 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다.
        soup = BeautifulSoup(res, "html.parser")
        # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정

        for j in range(20):
            try:
                soup2 = soup.find_all('div', class_="text")[j]
                soup3 = soup2.find('a')['href']
                params.append(soup3)
            except Exception:
                pass
        #print(params)
    return params

def fetch_list_url2():
    params2 = fetch_list_url()
    #print(params2)
    f = open(get_save_path(), 'w', encoding="utf-8")
    for i in params2:
        list_url2 = i
        url2 = urllib.request.Request(list_url2) # url 요청에 따른 http 통신 헤더값을 얻어낸다
        res2 = urllib.request.urlopen(url2).read().decode("utf-8")
        # 영어가 아닌 한글을 담아내기 위한 문자셋인 유니코드 문자셋을
        # 사용해서 html 문서와 html  문서내의 한글을 res 변수에 담는다.
        soup = BeautifulSoup(res2, "html.parser") # res html 문서를 BeautifulSoup 모듈을 사용해서 검색할수있도록 설정
        result = soup.find_all('div', id="article_body")[0]
        result2 = result.get_text(strip=True, separator='\n')
        f.write(result2)
    f.close()

fetch_list_url2()