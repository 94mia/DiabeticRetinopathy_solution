import http.server
import socketserver

from http.server import BaseHTTPRequestHandler
from http import HTTPStatus


from PIL import Image

from io import BytesIO

import io

PORT = 8002


import os
import math
from glob import glob
import imagehash


image_root = './zhizhen'


class ImageHTTPRequestHandler(BaseHTTPRequestHandler):

    """Simple HTTP request handler with GET and HEAD commands.

    This serves files from the current directory and any of its
    subdirectories.  The MIME type for files is determined by
    calling the .guess_type() method.

    The GET and HEAD requests are identical except that the HEAD
    request omits the actual contents of the file.

    """

    def __init__(self):
        print('initialized without params!!')

    def __init__(self, request, client_address, server):
        self.root = './kaggle'
        self.num_perfolder = 100
        self.kaggle_train_data_num = 35000
        self.kaggle_test_data_num = 58000
        print('ImageHTTPRequestHandler initialized!')
        super(ImageHTTPRequestHandler, self).__init__(request, client_address, server)


    def do_GET(self):
        print('Content type: {0}'.format(self.headers['Content-type']))
        if self.headers['Content-type'] == 'image/jpeg':
            self._classify()
        elif self.headers['Content-type'] == 'text/plain':
            cmd = self.headers['cmd']
            if cmd == 'get_kaggle_train_folder':
                self._get_kaggle_train_folder()
            elif cmd == 'get_kaggle_test_folder':
                self._get_kaggle_test_folder()
            elif cmd == 'get_kaggle_train_image':
                self._get_kaggle_train_image()
            elif cmd == 'get_kaggle_test_image':
                self._get_kaggle_test_image()

    def do_POST(self):
        print('Content type: {0}'.format(self.headers['Content-type']))
        if self.headers['Content-type'] == 'image/jpeg':
            self._classify()
        elif self.headers['Content-type'] == 'text/plain':
            cmd = self.headers['cmd']
            if cmd == 'get_kaggle_train_folder':
                self._get_kaggle_train_folder()
            elif cmd == 'get_kaggle_test_folder':
                self._get_kaggle_test_folder()
            elif cmd == 'get_kaggle_train_image':
                self._get_kaggle_train_image()
            elif cmd == 'get_kaggle_test_image':
                self._get_kaggle_test_image()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type, Accept, Content-Length")
        self.end_headers()

    def _get_kaggle_train_folder(self):
        print('begin to get kaggle train folder')
        folder_num = math.ceil(self.kaggle_train_data_num // self.num_perfolder)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", 'text/plain')
        self.send_header("cmd", 'get_kaggle_train_folder')
        self.send_header("folder_num", folder_num)
        self.end_headers()
        print('end to get kaggle train folder')

    def _get_kaggle_test_folder(self):
        print('begin to get kaggle test folder')
        folder_num = math.ceil(self.kaggle_test_data_num // self.num_perfolder)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", 'text/plain')
        self.send_header("cmd", 'get_kaggle_test_folder')
        self.send_header("folder_num", folder_num)
        self.end_headers()
        print('end to get kaggle test folder')

    def _get_kaggle_train_image(self):
        print('begin to get kaggle train image ')
        sub_idx = self.headers['sub_index']
        folder_index = self.headers['folder_index']
        # images_path = os.path.join(self.root, 'train/'+str(folder_index))
        # images_list = glob(os.path.join(images_path, '*.jpg'))
        # image = images_list[int(sub_idx)]
        image = '/home/weidong/code/dr/DiabeticRetinopathy_solution/data/dme/dme/33_dr_0_dme_0.jpg'
        data = open(image, 'rb').read()
        pil_img = Image.open(image)
        image_uid = imagehash.average_hash(pil_img)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", 'image/jpeg')
        self.send_header("cmd", 'get_kaggle_train_image')
        self.send_header("image_uid", str(image_uid))
        self.end_headers()
        self.wfile.write(data)
        print('end to get kaggle train image')


    def _get_kaggle_test_image(self):
        print('begin to get kaggle test image ')
        sub_idx = self.headers['sub_index']
        folder_index = self.headers['folder_index']
        # images_path = os.path.join(self.root, 'test/'+str(folder_index))
        # images_list = glob(os.path.join(images_path, '*.jpg'))
        # image = images_list[int(sub_idx)]
        image = '/home/weidong/code/dr/DiabeticRetinopathy_solution/data/dme/dme/34_dr_2_dme_1.jpg'
        data = open(image, 'rb').read()
        pil_img = Image.open(image)
        image_uid = imagehash.average_hash(pil_img)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", 'image/jpeg')
        self.send_header("cmd", 'get_kaggle_test_image')
        self.send_header("image_uid", str(image_uid))
        self.end_headers()
        self.wfile.write(data)
        print('end to get kaggle test image')

    def _store_kaggle_annotation_result(self):
        return 501


Handler = ImageHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()
