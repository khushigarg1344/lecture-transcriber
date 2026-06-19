set1=set([1,(5,7,2),"khushi",True,9.8])
set2={"khushi","btech",8.8,True}
for x in set1:
    print(x)
for y in set2:
    print(y)
print(set1.union(set2))
print(set1.intersection(set2))
print(set1.difference(set2)) 

set2.add((2,5))
print(set2)
set1.discard("khushi")
print(set1)
set3=set2.copy()
print(set3)
set2.clear() 
print(set2)

