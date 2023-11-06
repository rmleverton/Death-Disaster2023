# from bs4 import BeautifulSoup
# import urllib.request

# import requests
# import shutil

# url=('https://www.reuters.com/pictures/pictures-600-days-war-ukraine-2023-10-15/')
# html_page = urllib.request.urlopen(url)
# soup = BeautifulSoup(html_page, features="html.parser")

# main = soup.find(id = 'main-content')

# count = 0
# for img in main.findAll('img'):
#     assa=(img.get('src'))  
#     new_image=(url+assa)

#     count+=1

#     response = requests.get(assa).content#, stream=True)
#     #pic = urllib.urlopen(response)
#     with open('D:/Death&Disaster/Death-Disaster2023/images/ukraine/' + str(count) +'.jpg', 'w') as file:
#         file.write(response.read())

# print(count)
    

from bs4 import BeautifulSoup
import urllib.request
import requests

url = 'https://www.reuters.com/pictures/pictures-600-days-war-ukraine-2023-10-15/'
html_page = urllib.request.urlopen(url)
soup = BeautifulSoup(html_page, features="html.parser")

main = soup.find(id='main-content')

count = 0
# for img in main.findAll('img'):
#     img_src = img.get('src')
#     if img_src:
#         new_image_url = urllib.parse.urljoin(url, img_src)

#         count += 1

#         response = requests.get(new_image_url).content

#         with open(f'D:/Death&Disaster/Death-Disaster2023/images/ukraine/{count}.jpg', 'wb') as file:
#             file.write(response)

for caption in soup.findAll("div", class_="image-caption__caption__1npJ7"):
    text = caption.find("p", class_="text__text__1FZLe").get_text()
    count += 1
    print(text)

print(f"Downloaded {count} images.")
