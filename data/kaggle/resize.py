from PIL import Image
scale_sizes = [1024, 512, 256, 128]
#image_paths = ['train_images/' + line.strip() for line in open('train_images.txt', 'r')]
image_paths = ['val_images/' + line.strip() for line in open('val_images.txt', 'r')]
for index in range(len(image_paths)):
	image = Image.open(image_paths[index] + '.jpeg').convert('RGB')
	w, h = image.size
	tw, th = (min(w, h), min(w, h))
	image = image.crop((w // 2 - tw // 2, h // 2 - th // 2, w // 2 + tw // 2, h // 2 + th // 2))
	#image.save(image_paths[index] + '.png')
	w, h = image.size
	for scale_size in scale_sizes:
		tw, th = (scale_size, scale_size)
		ratio = tw / w
		assert ratio == th / h
		if ratio < 1:
			image = image.resize((tw, th), Image.ANTIALIAS)
		elif ratio > 1:
			image = image.resize((tw, th), Image.CUBIC)
		image.save(image_paths[index] + '_' + str(scale_size) + '.png')
