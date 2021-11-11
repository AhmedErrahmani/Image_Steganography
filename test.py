import os
import random
from PIL import Image
from steganography import Steganography
dir=os.listdir("./images")

s=Steganography()

# obj=s.merge(Image.open("res/img1.jpg"),Image.open("res/img2.jpg"))
# obj.save("output.jpg")
# for i in dir:
#     source=Image.open("./images/{}".format(i))
#
#     image_to_be_stgen=Image.open("./images/{}".format(random.choice(dir)))
#     try:
#         setg=s.merge(source,image_to_be_stgen)
#         source.save("./Dataset/0/{}".format(i))
#         setg.save("./Dataset/1/{}".format(i))
#     except:
#         print("image has been skipped due to error:",i)

#
# ans=s.unmerge(Image.open("output.jpg"))
# ans.save("origional.jpg")
img1=Image.open("./images/0.jpg")
map=img1.load()
print(map[4,5])
print(img1.load(),img1)