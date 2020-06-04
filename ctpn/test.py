import os
image_name ='G:\\ctpn\\data\\demo\\010.jpg'
base_name = image_name.split('\\')[-3]
s='data/results/' + 'res_{}.txt'.format(base_name.split('.')[0])

n=image_name.split('\\')
print(base_name)
