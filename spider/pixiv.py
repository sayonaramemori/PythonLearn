import requests
from header import headers,headers_for_img
from datetime import datetime
from download_img import download_imgs
import os

urls = []
ids = []

tags = ['Kikyo']
tag = tags[0]

os.mkdir(tag)
for i in range(1,8):
    url = f'https://www.pixiv.net/ajax/search/artworks/{tag}?word={tag}&order=date_d&mode=all&p={i}&csw=0&s_mode=s_tag&type=all&lang=zh'
    response = requests.get(url,headers=headers)
    # response = requests.get(url,headers=headers,proxies=proxy)
    res = response.json()
    data = res['body']['illustManga']['data']
    print(data)
    for item in data:
        id=item['id']
        ids.append(id)
        url=item['url']
        urls.append(url)
        print(f'{id} with {url}')
    response.close()

download_imgs(tag,urls,ids)
