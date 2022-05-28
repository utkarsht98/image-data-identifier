import os
count = 0

# print(os.listdir())
for i in os.listdir():
    os.rename(i,str(count)+'.'+i.split('.')[-1])
    count += 1