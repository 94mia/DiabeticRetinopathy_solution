import http.client
# conn = http.client.HTTPConnection('127.0.0.1', port=8002)

conn = http.client.HTTPConnection('yq01-idl-gpu-online5.yq01.baidu.com', port=8002)

# conn = http.client.HTTPConnection("face.baidu.com")

data = open('/Users/zhangweidong03/Code/dl/pytorch/github/dl_solution/dl_solution/1.jpg', 'rb').read()

headers = {"Content-type": "image/jpeg", "Accept": "q=0.6, image/jpeg", "Content-Length": str(len(data)), "algo":"zz"}

# conn.request('POST', "/test/for/medical", data, headers)
# conn.request('GET', "/test/for/medical", data, headers)
# conn.request('POST', "", data, headers)
conn.request('GET', "", data, headers)

r = conn.getresponse()

print('The image dr level is: {}'.format(r.headers['idx']))
print('The image dr level propobility is: {}'.format(r.headers['prop']))

image_uid = r.headers['image_uid']

print(image_uid)

headers = {"Content-type": "text/plain", "level":"3", "image_uid":image_uid}

conn.request('GET', "", "", headers=headers)
# conn.request('GET', "/test/for/medical", "", headers=headers)


# 这句似乎是必要的，'''r = conn.getresponse()'''，没有这句，服务器上不执行这一段
r = conn.getresponse()

print('end')

conn.close()