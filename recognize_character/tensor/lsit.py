from PIL import Image
im = Image.open('chinese/im_parts20.jpg')
reim = im.resize(17,17)
reim.show()