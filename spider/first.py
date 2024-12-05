from urllib.request import urlopen  

url = "https://www.pixiv.net/ajax/search/top/kafka?lang=zh&version=7771f749f08256057464c8e0c95738854a753080"

"""
response = urlopen(url)
content = response.read().decode('utf-8')
print(content)
response.close()
"""

proxy = {
    "https": "http://127.0.0.1:7899"
}


headers = {
    "accept": "application/json",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "zh-CN,zh;q=0.9",
    "baggage": "sentry-environment=production,sentry-release=7771f749f08256057464c8e0c95738854a753080,sentry-public_key=7b15ebdd9cf64efb88cfab93783df02a,sentry-trace_id=b2cfb75d70c546a08592ad86e432add1,sentry-sample_rate=0.0001",
    "cookie": "first_visit_datetime_pc=2024-12-02%2023%3A15%3A00; PHPSESSID=6mkc5m6g0ai24e79fj4r5krpg1igeic1; cc1=2024-12-02%2023%3A15%3A00; p_ab_id=3; p_ab_id_2=1; p_ab_d_id=1420001476; yuid_b=EUQFN4k; __cf_bm=jY8Xz61J11l.H5WjSDcRpEy8kb0ePneP2owAquyoR3Q-1733148900-1.0.1.1-cPehob74Is7zjEce00muK4FC6uNHvyPms2UYvsXuR68Bhm3dhjYN12flyw.RNUJrUuXs6yHWF9rjyCKgkR7wp1iAeYavBfV4N999mIJdaDo; __utma=235335808.1806695632.1733148902.1733148902.1733148902.1; __utmc=235335808; __utmz=235335808.1733148902.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utmt=1; cf_clearance=ByhQ_ZQyZu9hSI3UzVFLl8jpm0HlfnYShxRJxJbF6XY-1733148902-1.2.1.1-i1qdXiBKcx2oUXqRIzM.3MuSc7mvRl.jvMKeDx4kk.oJpUuAEYhlJ.TBmIo0ojDAdom7MpHXvvAZydAz0za8rfqWCyHMdcRBi8s5APNUdm23T_wlPgRgN.Q3jM9GMye8PGPv8lhZh0rTnZ5CYblDczGGqf6Povckv6ts32SMI5u74z3Zt.YbXfhBw9XhO79v1f.b3z.E.F2ednp4s7VBb_YLr5Fp4eMdaW5243ywOhnh1zB4k5M2raDzzDMRfVmfSrlN2aWleaH5wET7cNuTrkWlmJ6OW8A1HDSfIP38XpV2CtrBalAY.eyLc3DRG1Z597yaRmTC2EyAIGL9CERFqVBnBRaFKU.rL3c4sTekCTR8WUJaqra53xrSkvmpL0N3; _gid=GA1.2.1927053973.1733148912; __utmv=235335808.|2=login%20ever=no=1^3=plan=normal=1^9=p_ab_id=3=1^10=p_ab_id_2=1=1^11=lang=zh=1; _im_vid=01JE3VKQC8E4ME9J823VFN8H20; _ga_75BBYNYN9J=GS1.1.1733148903.1.1.1733149277.0.0.0; _ga=GA1.1.203329043.1733148904; __utmb=235335808.3.10.1733148902",
    "priority": "u=1, i",
    "referer": "https://www.pixiv.net/tags/kafka",
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "Windows",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "sentry-trace": "b2cfb75d70c546a08592ad86e432add1-a8bb0157060d0b43-0",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}
import requests
import re

headers={
    "Accept": "application/json",
    "Accept-encoding": "gzip, deflate, br, zstd",
    "Accept-language": "zh-CN,zh;q=0.9",
    "Baggage": "sentry-environment=production,sentry-release=2c2d77040749b81ba4e15128641940eab87b0552,sentry-public_key=7b15ebdd9cf64efb88cfab93783df02a,sentry-trace_id=6f2428176c6b4238945724b22f625dde,sentry-sample_rate=0.0001",
    "Cache-control": "no-cache",
    "Cookie": "first_visit_datetime_pc=2024-12-03%2012%3A18%3A52; PHPSESSID=mljq04n4fgvifomhefovlgq4s25pdk8e; cc1=2024-12-03%2012%3A18%3A52; p_ab_id=3; p_ab_id_2=7; p_ab_d_id=1098092522; yuid_b=I2ABRnM; __cf_bm=Ukb8K6d6PFUcgPsm.JQQ1p3cUlmaOkknKDDWgH_Uwao-1733195932-1.0.1.1-YTdYxuTrQn1B3hgOaVJqlIRTMY6HBu38VNcTiqBYMH6zXXFodCyvTKa39L_GRwi.dmGcVmW6utl5YdvQb6fR0.uMuHmrnVSOQhI__Hje9Jo; __utma=235335808.1285194609.1733195934.1733195934.1733195934.1; __utmc=235335808; __utmz=235335808.1733195934.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utmt=1; cf_clearance=CtBMuSnorbuzJJ08fS6xllfHuJmwGQQxBnmpngcF.Bg-1733195934-1.2.1.1-9zZ1BJruQW1BhhOW3n6OiqjEQs1nrbwgZh8fmqINLXWRNjHC4vKiP2b2yrNTJh22Je0q2db9211qjpqUyBdRtzoilJ8FSWghL1jWbh_8mXdfUuqzQHMY4DEUYo4M2BqQBJmUossxqEx0_LH_qF4qWtMWaaIzlZn660ZrYlyiVltvblmrlg7Byc1FSiDMWhvVlSCiI.ptA1aaFpf1Y9Bdroayn3JL_C8VHLbNWALXC8p5uQbP4pDWp3tl1NSD1zky1.mE_LgwMwSzP5diWUy_EbGqGB0Tz2znchYWgLwsd6rGa0MkGNvV2g0xc4_QzmSKSjyKs9_HORyR88b50NWgRIh1SqdbBG_NHtBEEn8dmbrUllcm0xfDB0VNCSDrMk1V; _gid=GA1.2.826447210.1733195935; __utmv=235335808.|2=login%20ever=no=1^3=plan=normal=1^9=p_ab_id=3=1^10=p_ab_id_2=7=1^11=lang=zh=1; _im_vid=01JE58EPY6ZQA8K6NXKDZM0YYF; _ga=GA1.1.1678854243.1733195934; _ga_75BBYNYN9J=GS1.1.1733195934.1.1.1733196392.0.0.0; __utmb=235335808.8.9.1733195935971",
    "Pragma": "no-cache",
    "Priority": "u=1, i",
    "Referer": "https://www.pixiv.net/tags/Kafka/illustrations?p=2",
    "Sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-ch-ua-mobile": "?0",
    "Sec-ch-ua-platform": "Windows",
    "Sec-fetch-dest": "empty",
    "Sec-fetch-mode": "cors",
    "Sec-fetch-site": "same-origin",
    "Sentry-trace": "6f2428176c6b4238945724b22f625dde-9848da847a99fcfd-0",
    "User-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

urls = []
ids = []

tags = ['%E7%9F%A5%E6%9B%B4%E9%B8%9F','Kafka']
tag = tags[0]
for i in range(2,20):
    url = f'https://www.pixiv.net/ajax/search/illustrations/{tag}?word={tag}&order=date_d&mode=all&p={i}&csw=0&s_mode=s_tag_full&type=illust_and_ugoira&lang=zh&version=2c2d77040749b81ba4e15128641940eab87b0552'
    response = requests.get(url,headers=headers,proxies=proxy)
    res = response.json()
#with open("res.json",mode="w",encoding='utf-8') as f:
#    f.write(str(res))
    data = res['body']['illust']['data']
    for item in data:
        id=item['id']
        ids.append(id)
        url=item['url']
        urls.append(url)
        print(f'{id} with {url}')

    response.close()

headers = {
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Priority": "u=1, i",
    "Referer": "https://www.pixiv.net/",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": "Windows",
    "Sec-Fetch-Dest": "image",
    "Sec-Fetch-Mode": "no-cors",
    "Sec-Fetch-Site": "cross-site",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

from datetime import datetime

# Send a GET request to the URL
for i in range(len(urls)):
    response = requests.get(urls[i],headers=headers)
# Check if the request was successful
    if response.status_code == 200:
        # Save the image to a file
        id = ids[i]
        with open(f"./zhigeng/{id}.jpg", "wb") as file:
            file.write(response.content)
        print(f"{datetime.now()}: Image {id} downloaded successfully!")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
