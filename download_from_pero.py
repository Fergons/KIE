import asyncio
from httpx import AsyncClient
from functools import partial
from bs4 import BeautifulSoup
import os
import time


async def get_list_of_files(session: AsyncClient, document_id):
    url = f"https://pero-ocr.fit.vutbr.cz/ocr/show_results/{document_id}"
    response  = await session.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Find all figure elements
    figures = soup.find_all('figure')
    # Extract data for each figure
    image_data = []

    #<figure class="figure m-1" data-document="f90af3bb-4a1a-4059-870b-3c1c6f5b4a6e" data-image="67590d51-db71-4019-8545-604249373bfd" data-index="499">
    # <img alt="499.jpg" class="figure-img img-thumbnail mx-auto d-block lazy-img" data-src="/document/get_image_preview/67590d51-db71-4019-8545-604249373bfd" style="height: 200px; min-width: 142.20183486238534px;"/>
    for figure in figures:
        data_image = figure['data-image']
        alt_text = figure.img['alt']
        if alt_text and data_image:
            image_data.append(
                (alt_text.split('.')[0],
                    data_image)
            )
    return image_data
        
async def download_alto_file(session: AsyncClient, file_id , file_name, save_dir):
    url = f"https://pero-ocr.fit.vutbr.cz/document/get_alto_xml/{file_id}"
    response = await session.get(url)
    with open(f"{save_dir}/{file_name}.xml", "wb") as f:
        f.write(response.content)
    print(f"File {save_dir}/{file_name} saved.")


async def main():
    save_dir = 'data/VHA/alto'
    os.makedirs(save_dir, exist_ok=True)    
    document_id = "f90af3bb-4a1a-4059-870b-3c1c6f5b4a6e"
    remember_token_cookie=  {"remember_token": "xxxxxx"} # get the token from loged in session
    async with AsyncClient(cookies=remember_token_cookie) as session:
        files = await get_list_of_files(session, document_id)
        jobs = [download_alto_file(session, file_id, file_name, save_dir) for file_name, file_id in files]
        # await asyncio.gather(*jobs)
        # not to overload the server, as I did the first time
        for job in jobs: 
            await job
            time.sleep(1) 

if __name__ == "__main__":
    asyncio.run(main())