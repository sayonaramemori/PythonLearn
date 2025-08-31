import threading
import requests
from header import headers_for_img
from datetime import datetime

def download_single_img(fold_name:str,url:str,id:str):
    response = requests.get(url,headers=headers_for_img)
    if response.status_code == 200:
        with open(f"./{fold_name}/{id}.jpg", "wb") as file:
            file.write(response.content)
        print(f"{datetime.now()}: Image {id} downloaded successfully!")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

def download_imgs(fold_name:str,urls:list[str],ids:list[str]):
    '''
    urls: List of the img url
    ids:  List of the name corresponding the img
    '''
    batch = 20
    for i in range(0,len(urls),batch):
        temp_thread = []
        for j in range(batch):
            idx = i + j
            t = threading.Thread(target=download_single_img, args=(fold_name,urls[idx],ids[idx],))
            temp_thread.append(t)
            t.start()
        for i in temp_thread:
            i.join()
