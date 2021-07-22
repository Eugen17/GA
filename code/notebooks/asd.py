from importlib import reload

# import emo as es

# ranks = []
# values = ([7793.0, 7554.0])
# ranks.append( es.calculate_rank(list(values)))
# print(ranks)

a = (10, 10,1)
b = (4, 4,100)
c = tuple(x-y for x, y in zip(a, b))
print(c)

l1 = [55,12,32,12,44,44,56]
def x (l):
    for i in reversed(range(len(l))):
        if l[i]>20:
            print(i)
            l.pop(i)
x(l1)

print(l1)
