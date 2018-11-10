import pandas as pd
data = pd.read_csv('/Users/siddhantbansal/Desktop/Project_files/HIT_results.csv')
url = data['Input.image_url']
name = []
for items in url:
    text = items.split('/')
    file = (text[-2])
    names = file.split('.')[0]
    name.append(names)
df = pd.DataFrame({'col': name})
df.to_csv('/Users/siddhantbansal/Desktop/names.csv', sep=',', index=False)
