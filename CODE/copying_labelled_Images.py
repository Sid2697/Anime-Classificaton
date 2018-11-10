# matching = [s for s in files if namea[0] in s]
matching = []
b = 0
for i in range(len(names)):
    try:
        a = [s for s in files if names[i] in s]
        matching.append(a)
    except:
        b += 1
