import os

path = 'C:/Users/Avalon/Desktop/Real_World_Snow'
file_list = os.listdir(path)

i = 0;

for item in file_list:
    src = os.path.join(path, item)
    dst = os.path.join(os.path.abspath(path), str(i)+'.png')

    try:
        os.rename(src, dst)
        i = i+1
    except Exception as e:
        print(e)
