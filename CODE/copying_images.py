import shutil
import pandas as pd

path = '/home/stu15/s15/ts6442/Capstone/images/images/'
name = pd.read_csv('/home/stu15/s15/ts6442/Capstone/codes/final_names.csv', header=None)
name = name.values.tolist()
final = []
for i in range(len(name)):
    final.append(name[i][0])

print(type(final))
print("Length of final list", len(final))

i = 0
for item in final:
    # print(i, item)
    i += 1
    try:
        shutil.copy2(path + item, '/home/stu15/s15/ts6442/Capstone/Labelled_images')
    except:
        print('[INFO] exception occured at', i)
        pass
