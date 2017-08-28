import http.client
from PIL import Image
from io import BytesIO

cmds = ['get_kaggle_train_folder', 'get_kaggle_test_folder', 'get_kaggle_train_image', 'get_kaggle_test_image']


conn = http.client.HTTPConnection('127.0.0.1', port=8002)


# interface 1: get kaggle train data folder number

headers = {"Content-type": "text/plain", "cmd":cmds[0]}

conn.request('GET', "", "", headers=headers)

r = conn.getresponse()

print(r.headers['folder_num'])


# interface 2: get kaggle test data number

headers = {"Content-type": "text/plain", "cmd":cmds[1]}

conn.request('GET', "", "", headers=headers)

r = conn.getresponse()

print(r.headers['folder_num'])


# interface 3: get kaggle train image by folder+index
headers = {"Content-type": "text/plain", "cmd":cmds[2], 'folder_index':'2', 'sub_index':'3'}
conn.request('GET', "", "", headers=headers)
r = conn.getresponse()

print('image uid: {}'.format(r.headers['image_uid']))

img = r.read()

img = BytesIO(img)

pil_img = Image.open(img)

pil_img.show()

# interface 3: get kaggle train image by folder+index
headers = {"Content-type": "text/plain", "cmd":cmds[2], 'folder_index':'2', 'sub_index':'3'}
conn.request('GET', "", "", headers=headers)
r = conn.getresponse()

print('image uid: {}'.format(r.headers['image_uid']))

img = r.read()

img = BytesIO(img)

pil_img = Image.open(img)

pil_img.show()


# interface 4: get kaggle train image by folder+index
headers = {"Content-type": "text/plain", "cmd":cmds[3], 'folder_index':'2', 'sub_index':'3'}
conn.request('GET', "", "", headers=headers)
r = conn.getresponse()

print('image uid: {}'.format(r.headers['image_uid']))

img = r.read()

img = BytesIO(img)

pil_img = Image.open(img)

pil_img.show()