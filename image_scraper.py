from bs4 import BeautifulSoup
from urllib.request import urlopen, urlretrieve
import urllib.parse
import os

def make_soup(url):
    html = urlopen(url).read()
    return BeautifulSoup(html, 'html.parser')

def sanitize_filename(filename):
    invalid_chars = ['?', '=', '&', '=']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def get_images(url):
    soup = make_soup(url)
    images = [img for img in soup.findAll('img')]
    print(str(len(images)) + " images found.")
    
    folder_name = 'roman'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")

    print(f'Downloading images to the "{folder_name}" folder.')

    image_links = [urllib.parse.urljoin(url, each.get('src')) for each in images]

    for each in image_links:
        filename = os.path.join(folder_name, sanitize_filename(each.split('/')[-1] + '.jpg'))
        urlretrieve(each, filename)

    print(f'Images saved in the "{folder_name}" folder.')
    return image_links

# A standard call looks like this
get_images('https://www.google.com/search?q=Roman+Theatre,+Amman&sca_esv=600400644&hl=ar&tbm=isch&sxsrf=ACQVn08Udreixit6rmzLm7SZtYd3lrfSYQ:1705945092341&source=lnms&sa=X&ved=2ahUKEwjWyvDZxPGDAxU_ywIHHRd2DAwQ_AUoAXoECAUQAw&biw=1366&bih=641&dpr=1')
