import http.server
import socketserver

from http.server import BaseHTTPRequestHandler
from http import HTTPStatus


from PIL import Image

from io import BytesIO

import io

PORT = 8002

from image_preprocessing import DrImageClassifier, get_kaggle_classifier, get_zz_classifier, get_all_classifier

# from utils import mdb
import imagehash
import os

kaggle_classifier = get_kaggle_classifier()
zz_classifier = get_zz_classifier()
all_classifier = get_all_classifier()

image_root = './zhizhen'

class ImageHTTPRequestHandler(BaseHTTPRequestHandler):

    """Simple HTTP request handler with GET and HEAD commands.

    This serves files from the current directory and any of its
    subdirectories.  The MIME type for files is determined by
    calling the .guess_type() method.

    The GET and HEAD requests are identical except that the HEAD
    request omits the actual contents of the file.

    """

    def do_GET(self):
        print('Content type: {0}'.format(self.headers['Content-type']))
        if self.headers['Content-type'] == 'image/jpeg':
            self._classify()

    # def do_POST(self):
    #     data1 = self.rfile.read(int(self.headers['Content-Length']))
    #     # print(self.rfile.read())
    #     # f=open('3.jpg','wb')
    #     # f.write(data1)
    #     stream = BytesIO(data1)
    #     img = Image.open(stream)
    #     # print(img.size())
    #     idx,prop,prop1 = classifier.classifyImage(img)
    #     print('dr image is level: {}'.format(idx))
    #     self.send_response(HTTPStatus.OK)
    #     self.send_header("Content-type", 'text/plain')
    #     self.send_header("idx", str(idx))
    #     self.send_header("prop", str(prop1))
    #     self.end_headers()

    def _classify(self):
        data1 = self.rfile.read(int(self.headers['Content-Length']))
        stream = BytesIO(data1)
        img = Image.open(stream)
        image_id = imagehash.average_hash(img)

        algo = self.headers['algo']
        classifier = kaggle_classifier
        if algo == 'kaggle':
            classifier = kaggle_classifier
        elif algo == 'zz':
            classifier = zz_classifier
        elif algo == 'all':
            classifier = all_classifier

        idx,prop= classifier.classifyImage(img)
        print(prop)

        print('dr image is level: {}'.format(idx))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", 'text/plain')
        self.send_header("idx", str(idx))
        self.send_header("prop", str(prop))
        self.send_header("image_uid", str(image_id))
        self.end_headers()


Handler = ImageHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()



# 问题解决：http://stackoverflow.com/questions/20689958/cannot-get-response-body-from-post-request-python
    # def do_POST(self):
    #     content = bytes("TEST RESPONSE", "UTF-8")
    #     self.send_response(200)
    #     self.send_header("Content-type", "text/plain")
    #     self.send_header("Content-Length", len(content))
    #     self.end_headers()
    #     print(self.rfile.read(int(self.headers['Content-Length'])).decode("UTF-8"))
    #     self.wfile.write(content)