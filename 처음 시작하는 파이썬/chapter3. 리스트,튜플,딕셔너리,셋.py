# 3.2 리스트
## 3.2.1 리스트 생성하기: [] 또는 list()
empty = []
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
big_birds = ['emu', 'ostrich', 'cassowary']
first_names = ['Graham', 'John', 'Terry', 'Terry', 'Michael']

another_empty_list = list()
print(another_empty_list)

## 3.2.2 다른 데이터 타입을 리스트로 변환하기: list()
list('cat')

a_tuple = ('ready', 'fire', 'aim')
print(a_tuple)
list(a_tuple)

birthday = '9/7/1991'
print(birthday.split('/'))

splitme = 'a/b//c/d///e'
print(splitme.split('/'))
print(splitme.split('//'))

## 3.2.3 [offset]으로 항목 얻기
marxes = ['Groucho', 'Chico', 'Harpo']
print(marxes[0], marxes[1], marxes[2])
print(marxes[-1], marxes[-2], marxes[-3])

## 3.2.4 리스트의 리스트
small_birds = ['hummingbird', 'finch']
extinct_birds = ['dodo', 'passenger pigeon', 'Norwegian Blue']
carol_birds = [3, 'French hens', 2, 'turtledoves']
all_birds = [small_birds, extinct_birds, 'macaw', carol_birds]
print(all_birds)
print(all_birds[0])
print(all_birds[1])
print(all_birds[1][0])

## 3.2.5 [offset]으로 항목 바꾸기
marxes = ['Groucho', 'Chico', 'Harpo']
print(marxes)
marxes[2] = 'Wanda'
print(marxes)

## 3.2.8 리스트 병합하기: extend() 또는 +=
marxes = ['Groucho', 'Chico', 'Harpo', 'Zeppo']
others = ['Gummo', 'Karl']
marxes.extend(others)
print(marxes)

marxes += others
print(marxes)

marxes.append(others)
print(marxes)

## 3.2.9 오프셋과 insert()로 항목 추가하기
marxes = ['Groucho', 'Chico', 'Harpo', 'Zeppo']
print(marxes)
marxes.insert(3, 'Gummo')
print(marxes)

marxes.insert(10, 'Karl')
print(marxes)

## 3.2.10 오프셋으로 항목 삭제하기: del
del marxes[-1]
print(marxes)

marxes = ['Groucho', 'Chico', 'Harpo', 'Gummo', 'Zeppo']
print(marxes[2])
del marxes[2]
print(marxes)
print(marxes[2])

## 3.2.11 값으로 항목 삭제하기: remove()
marxes = ['Groucho', 'Chico', 'Harpo', 'Gummo', 'Zeppo']
marxes.remove('Gummo')
print(marxes)

## 3.2.16 문자열로 변환하기: join()
marxes = ['Groucho', 'Chico', 'Harpo']
print(marxes)
', '.join(marxes)
print(marxes)

# join()은 split()의 반대이다. 증명하기
friends = ['Harry', 'Hermione', 'Ron']
separator = ' * '
joined = separator.join(friends)
print(joined)
separated = joined.split(separator)
print(separated)
print(separated == friends)

## 3.2.17 정렬하기: sort()
# sort()는 리스트 자체를 내부적으로 정렬한다.
# sorted()는 리스트의 정령된 복사본을 반환한다.
marxes = ['Groucho', 'Chico', 'Harpo']
sorted_marxes = sorted(marxes)
print(sorted_marxes)

marxes.sort()
print(marxes)

